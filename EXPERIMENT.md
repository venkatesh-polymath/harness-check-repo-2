# EXPERIMENT — g3_cnn

The experiment code is ALREADY committed at src/run_gen.py. Do NOT write or edit any
code. Your whole job is to run two Bash commands and report.

## Step 1 — sanity (fast, ~1 minute)
Run exactly:
    SH_EXP=g3_cnn SH_SMOKE=1 python src/run_gen.py
Confirm it prints "DONE" and exits 0. (It writes to results/*_smoke/ — ignore that dir.)
If it errors, print the traceback and STOP (do not attempt fixes).

## Step 2 — the real run (ONE blocking command; WAIT for it)
Run exactly this single command and wait until it returns:
    SH_EXP=g3_cnn python src/run_gen.py 2>&1 | tee results/g3_cnn/run.log
This writes results/g3_cnn/RESULTS.json incrementally and, on completion, a final
object with "status":"DONE". Do NOT run seeds yourself or supervise the loop
turn-by-turn — just let the one command run to completion.

## Finish
Print results/g3_cnn/RESULTS.json and stop. Never commit weights.
