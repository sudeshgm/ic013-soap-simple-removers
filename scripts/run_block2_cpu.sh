#!/usr/bin/env bash
# IC013 Block 2 reproduction gate: full ECSSD TokenCut via SOAP's main_salient.py (CPU copy from smoke_patch.py).
# Needs: /workspace/soap @06ede71 (+ python smoke_patch.py /workspace/soap), ECSSD in $ECSSD/{images,ground_truth_mask},
# fitted weights in $W. Outputs: $OUT/<run>/salient_segmentation_results.json
set -uo pipefail
ECSSD=${ECSSD:-/workspace/ic013_data/ecssd}; W=${W:-/workspace/ic013_data/weights/weights}; OUT=${OUT:-/workspace/ic013_data/block2}
cd ${SOAP_ROOT:-/workspace/soap}; export SMOKE_N=1000
run(){ name=$1; shift; mkdir -p $OUT/$name; echo "== $name $(date +%T)";
  python main_salient_cpu.py --datafolder $ECSSD --checkpoint_folder $W --dump_dir $OUT/$name --n_visualize 0 "$@" > $OUT/$name/log.txt 2>&1; tail -4 $OUT/$name/log.txt; }
run dinov2_R0            --backbone dinov2_base --features out
run dinov2_R1_mu2        --backbone dinov2_base --features out --semantic_invariance_projector
run dinov2_R1_mu2_guide  --backbone dinov2_base --features out --semantic_invariance_projector --pca_guide_dim -2 --flip_guide
run dinov2_R1_muSI075    --backbone dinov2_base --features out --semantic_invariance_projector --mu none
run mae_R0_k             --backbone mae_base --features k
run mae_R1_mu2_k         --backbone mae_base --features k --semantic_invariance_projector
run mae_R1_muSI075_k     --backbone mae_base --features k --semantic_invariance_projector --mu none
run mae_R0_out           --backbone mae_base --features out
run mae_R1_mu2_out       --backbone mae_base --features out --semantic_invariance_projector
