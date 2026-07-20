# EXPERIMENT — iv_sweep_K4
Code committed at src/run_intervention_sweep.py. Do NOT edit code.
Step1 sanity: `SH_K=4 SH_SMOKE=1 python src/run_intervention_sweep.py` (confirm DONE, exit0; ignore *_smoke dir). If error, print traceback and STOP.
Step2 real run (ONE blocking cmd, WAIT): `SH_K=4 python src/run_intervention_sweep.py 2>&1 | tee results/iv_sweep_K4/run.log`
Writes results/iv_sweep_K4/RESULTS.json (final status DONE). Do not supervise turn-by-turn.
Finish: print results/iv_sweep_K4/RESULTS.json. Never commit weights.