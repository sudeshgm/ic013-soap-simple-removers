"""IC013 Exp -1 (CPU): subspace overlap of SOAP's removed set vs top-k PCs vs INSID3 noise-SVD.
Inputs per backbone in --weights: <bb>_cov.pth, <bb>_agg_patch_responses{,_synth}.pth, <bb>_noise_feats.pt
SOAP weights are produced with SOAP's own SOAP.from_modelname using main_salient.py defaults
(score_version='scaled', gamma(alpha)=1.0, mu=2.0, tau=0.05). w_d is read back as diag(V^T (I-P) V).
Usage: python exp1.py --weights DIR --backbones dinov2_base ... --out results.json"""
import os, sys, json, argparse, math, torch
sys.path.insert(0, os.environ.get('SOAP_ROOT', '/workspace/soap'))
from soap import SOAP, WelfordChanEstimator
from soap.soap import semantic_invariance_score
torch.set_default_dtype(torch.float64)

def pa(A, B):
    s = torch.linalg.svdvals(A.T @ B).clamp(0, 1)
    return float((s**2).mean()), float(math.degrees(math.acos(float(s.min()))))

def rand_null(D, k, n=200, seed=0):
    g = torch.Generator().manual_seed(seed); m, l = [], []
    for _ in range(n):
        A = torch.linalg.qr(torch.randn(D, k, generator=g))[0]; B = torch.linalg.qr(torch.randn(D, k, generator=g))[0]
        a, b = pa(A, B); m.append(a); l.append(b)
    return sum(m)/n, sorted(l)[n//2]

def insid3(F, k):  # F: (P, D) uncentred, top-k right-singular vectors
    return torch.linalg.svd(F, full_matrices=False)[2][:k].T

def run(bb, wdir, mu, tau, gamma):
    est = WelfordChanEstimator.deserialize(os.path.join(wdir, f'{bb}_cov.pth'))
    cov = est.cov.double(); lam, V = torch.linalg.eigh(cov)  # ascending, same as SOAP
    D = len(lam)
    proj = SOAP.from_modelname(bb, wdir, alpha=gamma, mu=mu, tau=tau, score_version='scaled').projector.double()
    w = torch.diagonal(V.T @ (torch.eye(D) - proj) @ V)
    pd = torch.load(os.path.join(wdir, f'{bb}_agg_patch_responses.pth'), map_location='cpu').clamp(1e-12, 1-1e-12)
    ps = torch.load(os.path.join(wdir, f'{bb}_agg_patch_responses_synth.pth'), map_location='cpu').clamp(1e-12, 1-1e-12)
    si = semantic_invariance_score(pd, ps, alpha=gamma).double()  # eigh (ascending) order
    S = [d for d in range(D) if w[d] > 0.5]; k = len(S)
    r = dict(backbone=bb, D=D, k=k, S_rank=sorted(D-1-d for d in S),
             w_top10=sorted([round(float(x), 4) for x in w], reverse=True)[:10],
             si_top10_rank=[int(D-1-i) for i in torch.argsort(si, descending=True)[:10]],
             removed_var_frac=float((w*lam).sum()/lam.sum()))
    if k == 0: return r
    topk = list(range(D-k, D))
    VS, VT = V[:, S], V[:, topk]
    r['index_overlap'] = len(set(S) & set(topk)) / k
    r['weighted_overlap_trPP_over_k'] = float(w[topk].sum() / k)  # trace(V diag(w) V^T . V_k V_k^T)/k
    r['pa_topk_meancos2'], r['pa_topk_maxangle'] = pa(VS, VT)
    nf = torch.load(os.path.join(wdir, f'{bb}_noise_feats.pt')).double()
    r['pa_insid3_1img_meancos2'], r['pa_insid3_1img_maxangle'] = pa(VS, insid3(nf[0], k))
    r['pa_insid3_5img_meancos2'], r['pa_insid3_5img_maxangle'] = pa(VS, insid3(nf.reshape(-1, D), k))
    r['null_meancos2_k_over_D'] = k / D
    r['null_mc_meancos2'], r['null_mc_median_maxangle'] = rand_null(D, k)
    return r

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True); ap.add_argument('--backbones', nargs='+', required=True)
    ap.add_argument('--mu', type=float, default=2.0); ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--gamma', type=float, default=1.0); ap.add_argument('--out', default='exp1_results.json')
    a = ap.parse_args()
    res = [run(b, a.weights, a.mu, a.tau, a.gamma) for b in a.backbones]
    json.dump(res, open(a.out, 'w'), indent=1)
    for r in res: print(json.dumps(r))
