"""Operationalised 'strongest salient PC' rule (IC013; NOT in SOAP code, which hard-codes the dim in notebook.ipynb).
Uses only fit-time ImageNet artefacts (<bb>_agg_patch_responses{,_synth}.pth = per-rank fraction of patches with PC response>0).
Score_r = centre-bias(real map) - |centre-bias(synth map)| over the top-K ranked PCs, centre-bias = Pearson corr with a
centred Gaussian (sigma = grid/4). flip=True iff real centre-bias < 0 (guide = -response so the salient centre is argmax).
Prints guide index in main_salient's convention (negative index into ascending eigh)."""
import torch, sys, numpy as np
W = sys.argv[1]; bb = sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
L = lambda s: torch.load(f'{W}/{bb}_{s}.pth', map_location='cpu', weights_only=False).numpy()
R, S = L('agg_patch_responses'), L('agg_patch_responses_synth'); n, _, D = R.shape
yy, xx = np.mgrid[:n, :n]; c = (n - 1) / 2; g = np.exp(-((yy-c)**2 + (xx-c)**2) / (2*(n/4)**2)).ravel()
cb = lambda m: np.corrcoef(m.ravel(), g)[0, 1]
rows = [(r, cb(R[..., r]), cb(S[..., r])) for r in range(K)]
for r, a, b in rows: print(f'rank {r:2d} idx {-(r+1):3d}  real {a:+.3f}  synth {b:+.3f}  score {abs(a)-abs(b):+.3f}')
r, a, b = max(rows, key=lambda t: abs(t[1]) - abs(t[2]))
print(f'SELECTED {bb}: --pca_guide_dim {-(r+1)} {"--flip_guide" if a < 0 else ""}')
