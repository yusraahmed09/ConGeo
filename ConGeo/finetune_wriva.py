import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from congeo.dataset.wriva import WRIVADatasetTrain
from congeo.loss import InfoNCE
from congeo.model import TimmModel
from congeo.trainer import train


class ResizeToTensor:
    def __init__(self, img_size=384, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image=None):
        if image is None:
            raise ValueError('image is required')
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return {'image': img}


class SimpleConfig:
    def __init__(self, device, gpu_ids, verbose=True, clip_grad=0.0, scheduler='constant', normalize_features=True):
        self.device = device
        self.gpu_ids = gpu_ids
        self.verbose = verbose
        self.clip_grad = clip_grad
        self.scheduler = scheduler
        self.normalize_features = normalize_features


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms(img_size=384):
    return ResizeToTensor(img_size=img_size)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ConGeo on a WRIVA-style manifest.')
    parser.add_argument('--manifest-root', type=str, required=True, help='Directory containing manifest_train.csv, manifest_val.csv, manifest_test.csv or a single manifest.csv')
    parser.add_argument('--root-dir', type=str, required=True, help='Dataset root used to resolve relative paths in the manifest')
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='runs/wriva_finetune')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    manifest_root = Path(args.manifest_root)
    train_manifest = manifest_root / 'manifest_train.csv'
    val_manifest = manifest_root / 'manifest_val.csv'
    test_manifest = manifest_root / 'manifest_test.csv'
    single_manifest = manifest_root / 'manifest.csv'

    if train_manifest.exists() and val_manifest.exists() and test_manifest.exists():
        pass
    elif single_manifest.exists():
        train_manifest = single_manifest
        val_manifest = single_manifest
        test_manifest = single_manifest
    else:
        missing = [p for p in [train_manifest, val_manifest, test_manifest] if not p.exists()]
        raise FileNotFoundError(f'Missing manifests: {missing}')

    query_tfms = build_transforms(args.img_size)
    ref_tfms = build_transforms(args.img_size)

    train_ds = WRIVADatasetTrain(
        manifest_path=str(train_manifest),
        root_dir=args.root_dir,
        transforms_query=query_tfms,
        transforms_reference=ref_tfms,
    )
    val_ds = WRIVADatasetTrain(
        manifest_path=str(val_manifest),
        root_dir=args.root_dir,
        transforms_query=query_tfms,
        transforms_reference=ref_tfms,
    )
    test_ds = WRIVADatasetTrain(
        manifest_path=str(test_manifest),
        root_dir=args.root_dir,
        transforms_query=query_tfms,
        transforms_reference=ref_tfms,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = TimmModel(model_name=args.model_name, pretrained=True, img_size=args.img_size)
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = InfoNCE(torch.nn.CrossEntropyLoss(), device=args.device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = SimpleConfig(args.device, list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0], verbose=True)

    for epoch in range(args.epochs):
        loss = train(config, model, train_loader, loss_fn, optimizer, scheduler=scheduler, scaler=None)
        print(f'epoch={epoch + 1}/{args.epochs} train_loss={loss:.4f}')

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'wriva_finetuned.pt'))

    model.eval()
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for q, r, _ in test_loader:
            q = q.to(args.device)
            r = r.to(args.device)
            f1, f2 = model(q, r)
            logits = torch.matmul(f1, f2.t()) * model.logit_scale.exp() if not isinstance(model, torch.nn.DataParallel) else torch.matmul(f1, f2.t()) * model.module.logit_scale.exp()
            labels = torch.arange(q.size(0), device=args.device)
            loss = nn.functional.cross_entropy(logits, labels)
            test_loss += loss.item() * q.size(0)
            count += q.size(0)
    print(f'test_loss={test_loss / count:.4f}')


if __name__ == '__main__':
    main()
