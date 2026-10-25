# IC013 Block 1: Exp -1 (subspace overlap)
Scripts (scripts/): backbones.py (adds ibot_base, dinov3_base to SOAP's registry), make_imagenet_list.py (15/class, seed 0),
run_fit.py (SOAP's own fit_WCE+fit_projector imported unchanged, + 5 N(0,1) noise-image features, + optional fp16 cache),
exp1.py (CPU metrics), run_gpu.sh (end-to-end GPU driver), smoke_patch.py (CPU copies of SOAP scripts for the day-0 smoke test).
SOAP pinned at dsb-ifi/soap@06ede71. No datasets/features are committed (non-commercial licences).
Metrics: k=|{d: w_d>0.5}| with SOAP defaults from main_salient.py (scaled, gamma=1, mu=2, tau=0.05); index overlap |S∩topk|/k;
weighted overlap trace(V diag(w) V^T · P_topk)/k = sum_{d in topk} w_d / k; principal angles span(V_S) vs top-k and vs INSID3
(uncentred SVD of patch features of one / five N(0,1) input tensors, top-k right-singular vectors): mean cos², largest angle;
null k/D plus a 200-draw Monte-Carlo null (mean cos², median largest angle).
