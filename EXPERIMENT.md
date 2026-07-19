# EXPERIMENT — iv_smart

Code is ALREADY committed at src/run_intervention.py. Do NOT write or edit code.

## Step 1 — sanity (~1 min)
    SH_ARM=smart SH_SMOKE=1 python src/run_intervention.py
Confirm it prints "DONE" exit 0 (writes to results/*_smoke/, ignore). If it errors,
print the traceback and STOP.

## Step 2 — real run (ONE blocking command; WAIT for it to return)
    SH_ARM=smart python src/run_intervention.py 2>&1 | tee results/iv_smart/run.log
Writes results/iv_smart/RESULTS.json incrementally then a final "status":"DONE".
Do NOT supervise the loop turn-by-turn — let the one command finish.

## Finish
Print results/iv_smart/RESULTS.json and stop. Never commit weights.
