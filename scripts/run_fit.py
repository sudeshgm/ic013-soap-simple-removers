"""IC013 Exp -1 / full-test feature stage (GPU). For one backbone:
  1) SOAP's own fit_WCE + fit_projector (imported unchanged from dsb-ifi/soap fit_projector.py) on the 15k list
     -> <out>/<bb>_cov.pth, _agg_patch_responses.pth, _agg_patch_responses_synth.pth, _projector.pth
  2) INSID3 noise features: patch features of 5 N(0,1) input tensors (seed 0) -> <out>/<bb>_noise_feats.pt  (5,N,D) fp32
  3) optional fp16 patch-feature cache of the 15k images (deterministic Resize(256)+CenterCrop(224)) -> <cache>/<bb>_in15k_fp16.npy
Example: python run_fit.py --imagenet_train /data/imagenet/train --list in1k_train_15k.txt --backbone dinov2_base \
           --out weights --cache_dir feat_cache --device cuda --batch_size 128
"""
import os, sys, argparse, torch, numpy as np, torchvision as tv
from types import MethodType
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import get_backbone, SOAP_ROOT
sys.path.insert(0, SOAP_ROOT)
from fit_projector import fit_WCE, fit_projector
from soap.synth import SynthesizedDataSet

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class ListDS(torch.utils.data.Dataset):
    def __init__(s, root, files, proc): s.root, s.files, s.proc = root, files, proc
    def __len__(s): return len(s.files)
    def __getitem__(s, i): return s.proc(Image.open(os.path.join(s.root, s.files[i])).convert('RGB'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet_train', required=True); ap.add_argument('--list', required=True)
    ap.add_argument('--backbone', required=True); ap.add_argument('--out', default='weights')
    ap.add_argument('--cache_dir', default=None); ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=128); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None, help='smoke only')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = [l.strip() for l in open(a.list) if l.strip()][:a.limit]
    dev = torch.device(a.device); torch.manual_seed(0)
    imgsize = 224
    # identical to upstream fit_projector.py __main__ transform
    proc = tv.transforms.Compose([tv.transforms.RandomResizedCrop((imgsize, imgsize), (1., 1.)),
                                  tv.transforms.ToTensor(), tv.transforms.Normalize(MEAN, STD)])
    data = ListDS(a.imagenet_train, files, proc)
    dl = torch.utils.data.DataLoader(data, a.batch_size, num_workers=a.workers, pin_memory=(a.device == 'cuda'))
    dl_synth = SynthesizedDataSet(size=imgsize, channels=3, batch_size=a.batch_size, length=len(data))
    model = get_backbone(a.backbone)
    def forward_fn(self, img): return self.forward(img, 'out', -1)
    model.forward_fn = MethodType(forward_fn, model)
    model.to(dev).eval()
    cov = fit_WCE(model, dl, a.backbone, dev, forward_fn='forward_fn', num_global_tokens=0, dump_dir=a.out)
    fit_projector(model, cov, dl, dl_synth, a.backbone, imgsize, model.patch_size, dev,
                  forward_fn='forward_fn', num_global_tokens=0, dump_dir=a.out, soft_responses=False)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(5, 3, imgsize, imgsize, generator=g)
    with torch.no_grad():
        nf = torch.cat([model.forward_fn(noise[i:i+1].to(dev)).float().cpu() for i in range(5)])
    torch.save(nf, os.path.join(a.out, f'{a.backbone}_noise_feats.pt'))
    if a.cache_dir:
        os.makedirs(a.cache_dir, exist_ok=True)
        proc_eval = tv.transforms.Compose([tv.transforms.Resize(256), tv.transforms.CenterCrop(imgsize),
                                           tv.transforms.ToTensor(), tv.transforms.Normalize(MEAN, STD)])
        dl2 = torch.utils.data.DataLoader(ListDS(a.imagenet_train, files, proc_eval), a.batch_size, num_workers=a.workers)
        N = (imgsize // model.patch_size) ** 2
        mm = np.lib.format.open_memmap(os.path.join(a.cache_dir, f'{a.backbone}_in15k_fp16.npy'), 'w+', np.float16, (len(files), N, model.embed_dim))
        i = 0
        with torch.no_grad():
            for x in dl2:
                f = model.forward_fn(x.to(dev)).half().cpu().numpy(); mm[i:i+len(f)] = f; i += len(f)
        mm.flush()
    print('done', a.backbone)
