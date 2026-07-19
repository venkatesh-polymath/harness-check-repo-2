# EXPERIMENT — iv_triggered

The experiment code is ALREADY committed at src/run_intervention.py. Do NOT write or edit any
code. Your whole job is to run two Bash commands and report.

## Step 1 — sanity (fast, ~1 minute)
Run exactly:
    SH_ARM=triggered SH_SMOKE=1 python src/run_intervention.py
Confirm it prints "DONE" and exits 0. (It writes to results/*_smoke/ — ignore that dir.)
If it errors, print the traceback and STOP (do not attempt fixes).

## Step 2 — the real run (ONE blocking command; WAIT for it)
Run exactly this single command and wait until it returns:
    SH_ARM=triggered python src/run_intervention.py 2>&1 | tee results/iv_triggered/run.log
This writes results/iv_triggered/RESULTS.json incrementally and, on completion, a final
object with "status":"DONE". Do NOT run seeds yourself or supervise the loop
turn-by-turn — just let the one command run to completion.

## Finish
Print results/iv_triggered/RESULTS.json and stop. Never commit weights.
