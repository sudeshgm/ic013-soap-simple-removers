#!/usr/bin/env bash
# IC013 gate fix (a): re-fit SOAP projector for MAE on KEY features (--features k), same 15k list as Block 1
# (in1k_train_15k.txt, 15/class, seed 0; committed copy in results/ is used, else regenerated identically).
# Usage (Kaggle GPU, internet on): bash scripts/run_gpu_mae_k.sh   -> returns $WORK/ic013_weights_mae_k.tgz
set -euo pipefail
if [ -z "${IMAGENET_TRAIN:-}" ] || [ ! -d "$IMAGENET_TRAIN" ]; then
  IMAGENET_TRAIN=$(find /kaggle/input -maxdepth 7 -type d -path '*/Data/CLS-LOC/train' 2>/dev/null | head -1)
fi
[ -d "${IMAGENET_TRAIN:-}" ] || { echo "FATAL: ImageNet train dir not found"; ls -R /kaggle/input 2>/dev/null | head -40; exit 1; }
echo "Using IMAGENET_TRAIN=$IMAGENET_TRAIN"
WORK=${WORK:-$PWD/ic013_gpu_mae_k}; mkdir -p "$WORK"; cd "$WORK"
[ -d soap ] || { git clone https://github.com/dsb-ifi/soap && git -C soap checkout 06ede71; }
[ -d ic013 ] || git clone https://github.com/sudeshgm/ic013-soap-simple-removers ic013
python -m pip install -q torch torchvision "timm>=1.0.20" jaxtyping tqdm pillow numpy
export SOAP_ROOT=$WORK/soap
if [ -f ic013/results/in1k_train_15k.txt ]; then cp ic013/results/in1k_train_15k.txt .; else
  python ic013/scripts/make_imagenet_list.py "$IMAGENET_TRAIN" 15 > in1k_train_15k.txt; fi
wc -l in1k_train_15k.txt; md5sum in1k_train_15k.txt
python ic013/scripts/run_fit.py --imagenet_train "$IMAGENET_TRAIN" --list in1k_train_15k.txt \
   --backbone mae_base --features k --out weights_mae_k --device cuda --batch_size 128
tar czf ic013_weights_mae_k.tgz weights_mae_k in1k_train_15k.txt
ls -la ic013_weights_mae_k.tgz; echo "Return $WORK/ic013_weights_mae_k.tgz"
