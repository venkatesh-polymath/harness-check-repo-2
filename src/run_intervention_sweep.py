#!/usr/bin/env python3
"""
run_intervention_sweep.py — powered + budget-swept intervention (addresses the
5-seed / single-budget critique). Runs ALL five reset-timing arms at N_SEEDS=10
for a single matched budget K (env SH_K in {4,8,16}); one pod per K.
Validated regime: hidden=100 MLP, online Permuted-MNIST, SGD lr=0.10.
Writes results/iv_sweep_K<K>/RESULTS.json with per-(arm,seed) steady-state acc
and paired cross-arm contrasts (Wilcoxon).
"""
import os, sys, json, math, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim, torchvision
from scipy.stats import wilcoxon

K_EVENTS = int(os.environ.get("SH_K", "8"))
RESULTS_DIR = f"results/iv_sweep_K{K_EVENTS}"
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs("_weights", exist_ok=True)
RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, N_SEEDS, N_TASKS = 100, 10, 280
STEPS, BS, PROBE, NC, IND = 200, 128, 1000, 10, 784
LR, MOM, WD = 0.10, 0.9, 0.0
DATA_SEED, SS_WIN = 42, 50
TRIG_FRAC, TRIG_GAP, DEAD_STEP, DEAD_THRESH = 0.92, 8, 0.10, 0.50
ARMS = ["none", "triggered", "smart", "fixed", "random"]
if os.environ.get("SH_SMOKE") == "1":
    N_SEEDS, N_TASKS = 2, 20; RESULTS_DIR += "_smoke"; os.makedirs(RESULTS_DIR, exist_ok=True); RESULTS_PATH = os.path.join(RESULTS_DIR, "RESULTS.json")
print(f"SWEEP K={K_EVENTS} arms={ARMS} {N_SEEDS}seeds x {N_TASKS}tasks", flush=True)


class MLP(nn.Module):
    def __init__(s, h=100):
        super().__init__(); s.fc1=nn.Linear(IND,h); s.fc2=nn.Linear(h,h); s.head=nn.Linear(h,NC); s.relu=nn.ReLU()
    def forward(s,x): return s.head(s.relu(s.fc2(s.relu(s.fc1(x)))))
    def pen(s,x): h1=s.relu(s.fc1(x)); h2=s.relu(s.fc2(h1)); return h1,h2

def load():
    tr=torchvision.datasets.MNIST("/tmp/mnist",train=True,download=True); te=torchvision.datasets.MNIST("/tmp/mnist",train=False,download=True)
    return (tr.data.numpy().reshape(-1,IND).astype(np.float32)/255., tr.targets.numpy().astype(np.int64),
            te.data.numpy().reshape(-1,IND).astype(np.float32)/255., te.targets.numpy().astype(np.int64))
def perms(n,s=DATA_SEED):
    r=np.random.default_rng(s); p=[np.arange(IND)]
    for _ in range(n-1): p.append(r.permutation(IND))
    return p
def train(m,o,X,y,c,st,bs):
    m.train(); N=len(y); Xg=torch.from_numpy(X).to(DEVICE); yg=torch.from_numpy(y).to(DEVICE)
    idx=np.arange(N); np.random.shuffle(idx); pt=0
    for _ in range(st):
        if pt+bs>N: np.random.shuffle(idx); pt=0
        b=idx[pt:pt+bs]; pt+=bs; o.zero_grad(); c(m(Xg[b]),yg[b]).backward(); o.step()
@torch.no_grad()
def dmask(m,px,th=DEAD_THRESH):
    m.eval(); h1,h2=m.pen(px); return ((h1<=0).float().mean(0)>th).cpu().numpy(),((h2<=0).float().mean(0)>th).cpu().numpy()
@torch.no_grad()
def dfrac(m,px):
    m.eval(); h1,h2=m.pen(px); return ((( h1<=0).float().mean(0)>0.95).float().mean().item()+((h2<=0).float().mean(0)>0.95).float().mean().item())/2
@torch.no_grad()
def erank(m,px):
    m.eval(); _,h2=m.pen(px)
    try:
        S=torch.linalg.svdvals(h2.float()); S=S[S>1e-10]
        if len(S)==0: return 1.
        p=S/S.sum(); return max(1.,math.exp(-(p*torch.log(p+1e-12)).sum().item()))
    except: return 1.
@torch.no_grad()
def acc(m,X,y):
    m.eval(); Xg=torch.from_numpy(X).to(DEVICE); yg=torch.from_numpy(y).to(DEVICE); c=0
    for i in range(0,len(y),1024): c+=(m(Xg[i:i+1024]).argmax(1)==yg[i:i+1024]).sum().item()
    return c/len(y)
def probe(X,s=0):
    r=np.random.default_rng(s); return torch.from_numpy(X[r.choice(len(X),size=min(PROBE,len(X)),replace=False)]).to(DEVICE)
def rmom(o,p,rows=None,cols=None):
    b=o.state.get(p,{}).get("momentum_buffer")
    if b is None: return
    if rows is not None: b[rows]=0.
    if cols is not None: b[:,cols]=0.
@torch.no_grad()
def reset(m,o,px,gen):
    m1,m2=dmask(m,px); n=0
    i1=np.where(m1)[0]
    if len(i1):
        bd=1./math.sqrt(m.fc1.weight.shape[1])
        for j in i1: m.fc1.weight[j].uniform_(-bd,bd,generator=gen); m.fc1.bias[j].uniform_(-bd,bd,generator=gen); m.fc2.weight[:,j]=0.
        rmom(o,m.fc1.weight,rows=i1); rmom(o,m.fc1.bias,rows=i1); rmom(o,m.fc2.weight,cols=i1); n+=len(i1)
    i2=np.where(m2)[0]
    if len(i2):
        bd=1./math.sqrt(m.fc2.weight.shape[1])
        for j in i2: m.fc2.weight[j].uniform_(-bd,bd,generator=gen); m.fc2.bias[j].uniform_(-bd,bd,generator=gen); m.head.weight[:,j]=0.
        rmom(o,m.fc2.weight,rows=i2); rmom(o,m.fc2.bias,rows=i2); rmom(o,m.head.weight,cols=i2); n+=len(i2)
    return n
def fixed_tasks(k,n): st=n/(k+1); return set(int(round(st*(i+1))) for i in range(k))
def rand_tasks(k,n,s): return set(np.random.default_rng(1000+s).choice(np.arange(5,n),size=k,replace=False).tolist())

def run_arm(arm, seed, P, X_tr, y_tr, X_te, y_te, crit):
    torch.manual_seed(seed); np.random.seed(seed)
    gen=torch.Generator(device=DEVICE); gen.manual_seed(10000+seed)
    m=MLP(HIDDEN).to(DEVICE); o=optim.SGD(m.parameters(),lr=LR,momentum=MOM,weight_decay=WD)
    planned = fixed_tasks(K_EVENTS,N_TASKS) if arm=="fixed" else (rand_tasks(K_EVENTS,N_TASKS,seed) if arm=="random" else set())
    accs=[]; runmax=0.; last=-TRIG_GAP-1; last_dead=None; nev=0
    for t in range(N_TASKS):
        p=P[t]; train(m,o,X_tr[:,p],y_tr,crit,STEPS,BS)
        pr=probe(X_te[:,p],s=seed*10000+t); a=acc(m,X_te[:,p],y_te); er=erank(m,pr); df=dfrac(m,pr)
        accs.append(a); runmax=max(runmax,er)
        if last_dead is None: last_dead=df
        fire=False
        if arm=="triggered": fire = er<TRIG_FRAC*runmax and (t-last)>TRIG_GAP and nev<K_EVENTS and t>=5
        elif arm=="smart":   fire = (df-last_dead)>=DEAD_STEP and (t-last)>TRIG_GAP and nev<K_EVENTS and t>=5
        elif arm in ("fixed","random"): fire = t in planned
        if fire: reset(m,o,pr,gen); last=t; last_dead=df; nev+=1
    return dict(ss=float(np.mean(accs[-SS_WIN:])), task1=accs[0], final_dead=df, nev=nev)

def main():
    t0=time.time(); X_tr,y_tr,X_te,y_te=load(); crit=nn.CrossEntropyLoss(); P=perms(N_TASKS)
    data={a:{} for a in ARMS}
    for arm in ARMS:
        for seed in range(N_SEEDS):
            r=run_arm(arm,seed,P,X_tr,y_tr,X_te,y_te,crit); data[arm][seed]=r
            print(f"  K{K_EVENTS} {arm:9s} s{seed} ss={r['ss']:.4f} dead={r['final_dead']:.3f} nev={r['nev']} [{time.time()-t0:.0f}s]",flush=True)
        json.dump({"status":"RUNNING","K":K_EVENTS,"data":{a:{str(s):v for s,v in data[a].items()} for a in ARMS}}, open(RESULTS_PATH,"w"))
    def contrast(a,b):
        ss=sorted(set(data[a])&set(data[b])); d=np.array([data[a][s]['ss']-data[b][s]['ss'] for s in ss])
        try: p=float(wilcoxon(d).pvalue)
        except: p=float('nan')
        rng=np.random.default_rng(7); bt=[float(np.mean(rng.choice(d,len(d),replace=True))) for _ in range(5000)]
        return dict(mean_pp=float(d.mean()*100), ci95_pp=[float(np.percentile(bt,2.5)*100),float(np.percentile(bt,97.5)*100)], p=p, n=len(ss))
    contrasts={f"{a}_vs_none":contrast(a,"none") for a in ["triggered","smart","fixed","random"]}
    contrasts["triggered_vs_fixed"]=contrast("triggered","fixed"); contrasts["smart_vs_fixed"]=contrast("smart","fixed")
    means={a:float(np.mean([data[a][s]['ss'] for s in data[a]])) for a in ARMS}
    out={"status":"DONE","K_events":K_EVENTS,"n_seeds":N_SEEDS,"n_tasks":N_TASKS,
         "regime":"hidden=100 MLP online Permuted-MNIST SGD lr=0.10",
         "mean_ss_acc":means,
         "mean_final_dead":{a:float(np.mean([data[a][s]['final_dead'] for s in data[a]])) for a in ARMS},
         "mean_events":{a:float(np.mean([data[a][s]['nev'] for s in data[a]])) for a in ARMS},
         "contrasts":contrasts,
         "per_seed":{a:{str(s):data[a][s] for s in data[a]} for a in ARMS},
         "subject_executed":f"intervention budget sweep K={K_EVENTS}, {N_SEEDS} seeds, 5 arms",
         "metrics":{"K":K_EVENTS,"mean_ss_acc":means,"contrasts":{k:{'mean_pp':v['mean_pp'],'p':v['p']} for k,v in contrasts.items()}},
         "notes":"Powered (10-seed) matched-budget reset-timing comparison at K="+str(K_EVENTS)}
    json.dump(out,open(RESULTS_PATH,"w"),indent=2)
    print("DONE",json.dumps(out["metrics"],indent=2),flush=True)

if __name__=="__main__": main()
