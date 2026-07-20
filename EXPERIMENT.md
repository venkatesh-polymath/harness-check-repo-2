# EXPERIMENT — adam
Code committed at src/run_adam.py. Do NOT edit code.
Step1 sanity: ` SH_SMOKE=1 python src/run_adam.py` (confirm DONE, exit0; ignore *_smoke dir). If error, print traceback and STOP.
Step2 real run (ONE blocking cmd, WAIT): ` python src/run_adam.py 2>&1 | tee results/adam/run.log`
Writes results/adam/RESULTS.json (final status DONE). Do not supervise turn-by-turn.
Finish: print results/adam/RESULTS.json. Never commit weights.