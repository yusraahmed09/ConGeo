import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover
    rasterio = None

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


class WRIVADatasetTrain(Dataset):
    """Generic dataset for WRIVA-like ground/satellite pairs.

    The loader expects a manifest file (CSV/JSON/JSONL) with at least:
      - ground_path: path to the ground-view JPG
      - sat_path or sat_crop_path: path to the satellite image (.tif, .png, .jpg)

    Optional columns:
      - label: int/str identifier for the sample
      - site_id / weather_id / split
      - lat / lon: GPS coordinates used to crop from a GeoTIFF if sat_crop_path is absent
      - crop_x / crop_y / crop_w / crop_h: explicit pixel crop coordinates
      - crop_size: size for a square crop, e.g. 384
      - root_dir: optional base directory for relative paths

    Notes:
      - If a satellite image is mostly blank/white, the sample is skipped.
      - If a GeoTIFF is provided and lat/lon are present, the loader will try to
        crop a centered patch; if that fails, it will fall back to the whole raster.
    """

    def __init__(
        self,
        manifest_path: str,
        root_dir: Optional[str] = None,
        transforms_query=None,
        transforms_reference=None,
        prob_flip: float = 0.0,
        prob_rotate: float = 0.0,
        blank_threshold: float = 0.97,
        blank_rgb_threshold: int = 250,
        max_cache_size: int = 256,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.root_dir = root_dir
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.blank_threshold = blank_threshold
        self.blank_rgb_threshold = blank_rgb_threshold
        self.max_cache_size = max_cache_size
        self._cache: Dict[int, tuple] = {}
        self.samples = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        manifest_path = str(self.manifest_path)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        suffix = Path(manifest_path).suffix.lower()

        if suffix == ".csv":
            with open(manifest_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif suffix == ".jsonl":
            with open(manifest_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            with open(manifest_path, encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload if isinstance(payload, list) else payload.get("rows", [])
        else:
            raise ValueError("Manifest must be .csv, .json, or .jsonl")

        valid_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not row.get("ground_path"):
                continue
            if not (row.get("sat_path") or row.get("sat_crop_path") or row.get("satellite_path")):
                continue
            row = dict(row)
            row["ground_path"] = self._resolve_path(row.get("ground_path"))
            if row.get("sat_crop_path"):
                row["sat_crop_path"] = self._resolve_path(row.get("sat_crop_path"))
            if row.get("sat_path"):
                row["sat_path"] = self._resolve_path(row.get("sat_path"))
            if row.get("satellite_path"):
                row["satellite_path"] = self._resolve_path(row.get("satellite_path"))
            if self._is_missing(row["ground_path"]):
                continue
            ref_path = row.get("sat_crop_path") or row.get("sat_path") or row.get("satellite_path")
            if self._is_missing(ref_path):
                continue
            if self._is_blank_reference(ref_path, row):
                continue
            valid_rows.append(row)
        return valid_rows

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        path = str(path)
        if os.path.isabs(path):
            return path
        if self.root_dir:
            return os.path.join(self.root_dir, path)
        return path

    def _is_missing(self, path: Optional[str]) -> bool:
        return not path or not os.path.exists(path)

    def _is_blank_reference(self, ref_path: Optional[str], row: Dict[str, Any]) -> bool:
        try:
            img = self._load_reference_image(ref_path, row, fast_check=True)
        except Exception:
            return True
        if img is None:
            return True
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blank_ratio = (gray >= self.blank_rgb_threshold).mean()
        return bool(blank_ratio > self.blank_threshold)

    def _load_ground_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read ground image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_reference_image(self, path: str, row: Dict[str, Any], fast_check: bool = False) -> np.ndarray:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Could not read reference image: {path}")

        # If a pre-cropped PNG/JPG exists, use it directly.
        suffix = str(path).lower()
        if suffix.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read reference image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # If the reference is a GeoTIFF, try to crop using explicit pixel window.
        if suffix.endswith((".tif", ".tiff")):
            if rasterio is not None:
                with rasterio.open(path) as src:
                    img = self._read_rasterio(src, row)
                    if fast_check:
                        return img
                    if img is None:
                        raise FileNotFoundError(f"Could not decode raster: {path}")
                    return img
            if tifffile is not None:
                img = tifffile.imread(path)
                img = self._convert_tiff_to_rgb(img)
                if fast_check:
                    return img
                return img
            raise RuntimeError("Neither rasterio nor tifffile is available")

        # Fallback for generic image formats.
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read reference image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_reference_gps(self, row: Dict[str, Any]) -> Optional[Dict[str, float]]:
        # 1) manifest-level lat/lon columns
        lat = row.get("lat")
        lon = row.get("lon")
        if lat is not None and lon is not None:
            return {"lat": float(lat), "lon": float(lon)}

        # 2) reference JSON sidecar if present
        ref_json_path = row.get("reference_json") or row.get("reference_path")
        if ref_json_path and os.path.exists(ref_json_path):
            try:
                with open(ref_json_path, encoding="utf-8") as f:
                    payload = json.load(f)
                extrinsics = payload.get("extrinsics", {})
                lat = extrinsics.get("lat")
                lon = extrinsics.get("lon")
                if lat is not None and lon is not None:
                    return {"lat": float(lat), "lon": float(lon)}
            except Exception:
                pass

        # 3) fallback to ground image path metadata if present as a JSON file next to it
        ground_path = row.get("ground_path")
        if ground_path and os.path.exists(ground_path.replace(".jpg", ".json")):
            try:
                with open(ground_path.replace(".jpg", ".json"), encoding="utf-8") as f:
                    payload = json.load(f)
                extrinsics = payload.get("extrinsics", {})
                lat = extrinsics.get("lat")
                lon = extrinsics.get("lon")
                if lat is not None and lon is not None:
                    return {"lat": float(lat), "lon": float(lon)}
            except Exception:
                pass
        return None

    def _read_rasterio(self, src, row: Dict[str, Any]) -> np.ndarray:
        # First, try explicit crop window from the manifest.
        crop_x = row.get("crop_x")
        crop_y = row.get("crop_y")
        crop_w = row.get("crop_w")
        crop_h = row.get("crop_h")
        if crop_x is not None and crop_y is not None and crop_w is not None and crop_h is not None:
            window = Window(int(crop_x), int(crop_y), int(crop_w), int(crop_h))
            data = src.read(window=window)
            return self._arr_to_rgb(data)

        crop_size = row.get("crop_size")
        if crop_size is not None:
            gps = self._read_reference_gps(row)
            if gps is not None:
                try:
                    row_idx, col_idx = src.index(float(gps["lon"]), float(gps["lat"]))
                    h, w = int(crop_size), int(crop_size)
                    window = Window(max(0, col_idx - w // 2), max(0, row_idx - h // 2), w, h)
                    data = src.read(window=window)
                    return self._arr_to_rgb(data)
                except Exception:
                    pass

        # Fallback to the whole raster.
        data = src.read()
        return self._arr_to_rgb(data)

    def _convert_tiff_to_rgb(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4):
                img = np.moveaxis(img, 0, -1)
            elif img.shape[-1] in (1, 3, 4):
                img = img
            else:
                img = img[..., :3]
        return img.astype(np.uint8)

    def _arr_to_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return None
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[-1] in (1, 3, 4):
                arr = arr
            else:
                arr = arr[..., :3]
        return arr.astype(np.uint8)

    def __getitem__(self, index: int):
        if index in self._cache:
            return self._cache[index]

        row = self.samples[index]
        query_img = self._load_ground_image(row["ground_path"])
        ref_path = row.get("sat_crop_path") or row.get("sat_path") or row.get("satellite_path")
        reference_img = self._load_reference_image(ref_path, row)

        # Skip obvious blank/white tiles.
        gray = cv2.cvtColor(reference_img, cv2.COLOR_RGB2GRAY)
        blank_ratio = (gray >= self.blank_rgb_threshold).mean()
        if blank_ratio > self.blank_threshold:
            raise RuntimeError(f"Reference image is blank/near-white: {ref_path}")

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            c, h, w = query_img.shape
            shifts = -w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)

        label = row.get("label")
        if label is None:
            label = index
        try:
            label = int(label)
        except Exception:
            label = str(label)

        sample = (query_img, reference_img, torch.tensor(label, dtype=torch.long))
        if self.max_cache_size > 0:
            self._cache[index] = sample
            if len(self._cache) > self.max_cache_size:
                self._cache.pop(next(iter(self._cache)))
        return sample

    def __len__(self) -> int:
        return len(self.samples)
