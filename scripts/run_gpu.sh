#!/usr/bin/env bash
# IC013 Block 1 GPU stage. Needs: CUDA GPU, ImageNet-1k train folder (class subdirs), internet (timm/HF, torch.hub CAPI, iBOT ckpt).
# Usage: IMAGENET_TRAIN=/path/to/imagenet/train bash scripts/run_gpu.sh
set -euo pipefail
: "${IMAGENET_TRAIN:?set IMAGENET_TRAIN}"
WORK=${WORK:-$PWD/ic013_gpu}; mkdir -p "$WORK"; cd "$WORK"
[ -d soap ] || git clone https://github.com/dsb-ifi/soap && git -C soap checkout 06ede71
[ -d ic013 ] || git clone https://github.com/sudeshgm/ic013-soap-simple-removers ic013
python -m pip install -q torch torchvision "timm>=1.0.20" jaxtyping tqdm pillow numpy
export SOAP_ROOT=$WORK/soap
python ic013/scripts/make_imagenet_list.py "$IMAGENET_TRAIN" 15 > in1k_train_15k.txt
wc -l in1k_train_15k.txt
for BB in dinov2_base ibot_base mae_base capi_large dinov3_base; do
  BS=128; [ "$BB" = capi_large ] && BS=64
  python ic013/scripts/run_fit.py --imagenet_train "$IMAGENET_TRAIN" --list in1k_train_15k.txt \
     --backbone $BB --out weights --cache_dir feat_cache --device cuda --batch_size $BS
done
python ic013/scripts/exp1.py --weights weights --backbones dinov2_base ibot_base mae_base capi_large dinov3_base --out exp1_results.json
tar czf ic013_weights.tgz weights in1k_train_15k.txt exp1_results.json   # small (~10 MB/backbone); feat_cache stays on GPU host
echo "Return ic013_weights.tgz; keep feat_cache/*.npy (fp16, ~6-8 GB each) for the full test."
