# EXPERIMENT — iv_K8_fixed
src/run_intervention_sweep.py is committed. Do NOT edit code.
Step1 sanity: `SH_K=8 SH_ARM=fixed SH_SMOKE=1 python src/run_intervention_sweep.py` (expect DONE exit0; ignore *_smoke). If error, print traceback + STOP.
Step2 (ONE blocking cmd, WAIT for full completion): `SH_K=8 SH_ARM=fixed python src/run_intervention_sweep.py 2>&1 | tee results/iv_K8_fixed/run.log`
It writes results/iv_K8_fixed/RESULTS.json ending status DONE. This takes ~20 min — WAIT the whole time, do not stop early.
Finish: print results/iv_K8_fixed/RESULTS.json.