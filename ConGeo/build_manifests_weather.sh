python build_wriva_manifest_optimized.py \
  --root /squash/yu395012-wriva-cvgl-baseline-cvgl_224/ \
  --out /home/yu395012/ConGeo/ConGeo/manifests_weather_split/manifest.csv \
  --split \
  --max-ground-per-site 1000 \
  --max-variants-per-site 8 \
  --max-pairs-per-site 8000