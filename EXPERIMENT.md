# EXPERIMENT — eos
Code is ALREADY committed at src/run_eos.py. Do NOT write or edit code.
## Step 1 sanity (~1 min): run  `SH_SMOKE=1 python src/run_eos.py`  — confirm it prints DONE exit 0 (writes results/eos_smoke/, ignore). If it errors, print traceback and STOP.
## Step 2 real run (ONE blocking command, WAIT):  `python src/run_eos.py 2>&1 | tee results/eos/run.log`  — writes results/eos/RESULTS.json incrementally then final status DONE. Do NOT supervise turn-by-turn.
## Finish: print results/eos/RESULTS.json and stop. Never commit weights.