"""Deterministic 15k ImageNet-train subset: 15 images per class (1000 classes), seed 0.
Usage: python make_imagenet_list.py /path/to/imagenet/train > in1k_train_15k.txt (paths relative to train/)."""
import os, sys, random
root = sys.argv[1]; per = int(sys.argv[2]) if len(sys.argv) > 2 else 15
rng = random.Random(0)
for c in sorted(os.listdir(root)):
    fs = sorted(os.listdir(os.path.join(root, c)))
    for f in rng.sample(fs, min(per, len(fs))): print(f'{c}/{f}')
