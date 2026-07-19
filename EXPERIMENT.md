# EXPERIMENT — g1_fashion

Code is ALREADY committed at src/run_gen.py. Do NOT write or edit code.

## Step 1 — sanity (~1 min)
    SH_EXP=g1_fashion SH_SMOKE=1 python src/run_gen.py
Confirm it prints "DONE" exit 0 (writes to results/*_smoke/, ignore). If it errors,
print the traceback and STOP.

## Step 2 — real run (ONE blocking command; WAIT for it to return)
    SH_EXP=g1_fashion python src/run_gen.py 2>&1 | tee results/g1_fashion/run.log
Writes results/g1_fashion/RESULTS.json incrementally then a final "status":"DONE".
Do NOT supervise the loop turn-by-turn — let the one command finish.

## Finish
Print results/g1_fashion/RESULTS.json and stop. Never commit weights.
