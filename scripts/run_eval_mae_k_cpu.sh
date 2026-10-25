#!/usr/bin/env bash
# IC013 gate fix (b): MAE R1 on full ECSSD with the key-feature projector (from run_gpu_mae_k.sh).
# Always runs vmax (info). Guided run only if GUIDE_DIM is set (e.g. GUIDE_DIM=-4 FLIP=1); SOAP gives no rule-derived
# MAE guide (see GUIDE_RULE.md), so do not set it by looking at ECSSD scores.
set -uo pipefail
ECSSD=${ECSSD:-/workspace/ic013_data/ecssd}; W=${W:-/workspace/ic013_data/weights_mae_k/weights_mae_k}
OUT=${OUT:-/workspace/ic013_data/block2b}; mkdir -p $OUT; cd ${SOAP_ROOT:-/workspace/soap}; export SMOKE_N=1000
for f in cov agg_patch_responses agg_patch_responses_synth; do [ -f $W/mae_base_$f.pth ] || { echo "missing $W/mae_base_$f.pth"; exit 1; }; done
run(){ n=$1; shift; mkdir -p $OUT/$n; echo "== $n $(date +%T)"
  python main_salient_cpu.py --datafolder $ECSSD --checkpoint_folder $W --dump_dir $OUT/$n --n_visualize 0 \
    --backbone mae_base --features k --semantic_invariance_projector "$@" > $OUT/$n/log.txt 2>&1; tail -2 $OUT/$n/log.txt; }
python /workspace/ic013/scripts/salient_pc_rule.py $W mae_base 8 | tee $OUT/rule_output.txt
run mae_R1k_mu2_vmax
[ -n "${GUIDE_DIM:-}" ] && run mae_R1k_mu2_guide${GUIDE_DIM}${FLIP:+_flip} --pca_guide_dim $GUIDE_DIM ${FLIP:+--flip_guide}
python /workspace/ic013/scripts/summarize_block2.py $OUT /workspace/ic013/results/block2b_mae_k_ecssd.json
