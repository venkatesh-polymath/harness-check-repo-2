#!/usr/bin/env python3
"""
run_adam.py — precedence + predictive-AUC under ADAM (addresses the SGD-only
scope critique). Same regime + analysis as run_valprec.py, but optimizer=Adam
(lr=1e-3). Reports whether the network still loses plasticity under Adam, whether
dead-unit/effective-rank still predict, whether GNS still fails, AND the
monotone-null AUC (task index) so the signal margin over monotonicity is explicit.
Writes results/adam/RESULTS.json.
"""
import os, sys, json, math, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim, torchvision
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

_SUF = "_smoke" if os.environ.get("SH_SMOKE")=="1" else ""
RD = f"results/adam{_SUF}"; os.makedirs(RD,exist_ok=True); os.makedirs("_weights",exist_ok=True)
RP = os.path.join(RD,"RESULTS.json"); TRAJ=os.path.join(RD,"trajectories.json")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN,N_SEEDS,N_TASKS=100,8,300
STEPS,BS,PROBE,NC,IND=200,128,1000,10,784
ADAM_LR=1e-3; DATA_SEED=42; THR_PP=20.; MINT=2; K=5; GB,GBS=20,64
if os.environ.get("SH_SMOKE")=="1": N_SEEDS,N_TASKS=1,6
print(f"ADAM precedence hidden={HIDDEN} lr={ADAM_LR} {N_SEEDS}seeds x {N_TASKS}tasks",flush=True)

class MLP(nn.Module):
    def __init__(s,h=100):
        super().__init__(); s.fc1=nn.Linear(IND,h); s.fc2=nn.Linear(h,h); s.head=nn.Linear(h,NC); s.relu=nn.ReLU()
    def forward(s,x): return s.head(s.relu(s.fc2(s.relu(s.fc1(x)))))
    def pen(s,x): h1=s.relu(s.fc1(x)); h2=s.relu(s.fc2(h1)); return h1,h2

def load():
    tr=torchvision.datasets.MNIST("/tmp/mnist",train=True,download=True); te=torchvision.datasets.MNIST("/tmp/mnist",train=False,download=True)
    return (tr.data.numpy().reshape(-1,IND).astype(np.float32)/255.,tr.targets.numpy().astype(np.int64),
            te.data.numpy().reshape(-1,IND).astype(np.float32)/255.,te.targets.numpy().astype(np.int64))
def perms(n):
    r=np.random.default_rng(DATA_SEED); p=[np.arange(IND)]
    for _ in range(n-1): p.append(r.permutation(IND))
    return p
def train(m,o,X,y,c):
    m.train(); N=len(y); Xg=torch.from_numpy(X).to(DEVICE); yg=torch.from_numpy(y).to(DEVICE)
    idx=np.arange(N); np.random.shuffle(idx); pt=0
    for _ in range(STEPS):
        if pt+BS>N: np.random.shuffle(idx); pt=0
        b=idx[pt:pt+BS]; pt+=BS; o.zero_grad(); c(m(Xg[b]),yg[b]).backward(); o.step()
@torch.no_grad()
def dead(m,px):
    m.eval(); h1,h2=m.pen(px); return ((((h1<=0).float().mean(0)>0.95).float().mean().item())+(((h2<=0).float().mean(0)>0.95).float().mean().item()))/2
@torch.no_grad()
def erank(m,px):
    m.eval(); _,h2=m.pen(px)
    try:
        S=torch.linalg.svdvals(h2.float()); S=S[S>1e-10]
        if len(S)==0: return 1.
        p=S/S.sum(); return max(1.,math.exp(-(p*torch.log(p+1e-12)).sum().item()))
    except: return 1.
def gns(m,X,y,c):
    m.train(); N=len(y)
    if N<GB*GBS: return float('nan')
    idx=np.random.permutation(N); Xg=torch.from_numpy(X).to(DEVICE); yg=torch.from_numpy(y).to(DEVICE); gs=[]
    for i in range(GB):
        bi=idx[i*GBS:(i+1)*GBS]; m.zero_grad(); c(m(Xg[bi]),yg[bi]).backward()
        gs.append(torch.cat([p.grad.detach().flatten() for p in m.parameters() if p.grad is not None]))
    m.zero_grad(); G=torch.stack(gs); gm=G.mean(0); tr=(G-gm).pow(2).sum(1).sum().item()/(GB-1); sig=gm.pow(2).sum().item()
    return float(tr/sig) if sig>1e-20 and math.isfinite(tr) and tr/sig>0 else float('nan')
@torch.no_grad()
def wdrift(m,init):
    d=[abs(p.data.norm(2).item()/w-1.) for (n,p),w in zip(m.named_parameters(),init.values()) if w>1e-12]
    return float(np.mean(d)) if d else float('nan')
@torch.no_grad()
def acc(m,X,y):
    m.eval(); Xg=torch.from_numpy(X).to(DEVICE); yg=torch.from_numpy(y).to(DEVICE); c=0
    for i in range(0,len(y),1024): c+=(m(Xg[i:i+1024]).argmax(1)==yg[i:i+1024]).sum().item()
    return c/len(y)
def probe(X,s): r=np.random.default_rng(s); return torch.from_numpy(X[r.choice(len(X),size=min(PROBE,len(X)),replace=False)]).to(DEVICE)
def collapse(accs):
    if len(accs)<MINT+1: return None
    ref=accs[0]; cnt=0; first=None
    for t,a in enumerate(accs):
        if (ref-a)*100>=THR_PP:
            cnt+=1; first=first if first is not None else t
        else: cnt=0; first=None
        if cnt>=MINT: return first
    return None
def onset(v,frac=0.5,w=2):
    ma=[float(np.mean([x for x in v[max(0,i-w+1):i+1] if math.isfinite(x)])) if any(math.isfinite(x) for x in v[max(0,i-w+1):i+1]) else float('nan') for i in range(len(v))]
    cl=[(i,x) for i,x in enumerate(ma) if math.isfinite(x)]
    if len(cl)<4: return None
    v0,vf=cl[0][1],cl[-1][1]
    if abs(vf-v0)<1e-10: return None
    thr=v0+frac*(vf-v0); d=1 if vf>v0 else -1
    for i,x in cl:
        if d==1 and x>=thr: return i
        if d==-1 and x<=thr: return i
    return None
def pauc(recs,key,inv,taskidx=False):
    f,l=[],[]
    for r in recs:
        a=r['accs']; thr=a[0]-THR_PP/100; T=len(a); vals=list(range(T)) if taskidx else r[key]
        for t in range(T-1):
            v=vals[t]
            if v is None or (isinstance(v,float) and not math.isfinite(v)): continue
            lab=int(any(a[tt]<thr for tt in range(t+1,min(t+K+1,T)))); f.append(-v if inv else v); l.append(lab)
    if len(l)<10 or len(set(l))<2: return float('nan')
    try: return float(roc_auc_score(np.array(l),np.array(f,float)))
    except: return float('nan')

def main():
    t0=time.time(); X_tr,y_tr,X_te,y_te=load(); crit=nn.CrossEntropyLoss(); P=perms(N_TASKS)
    recs={}; trajs={}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        m=MLP(HIDDEN).to(DEVICE); o=optim.Adam(m.parameters(),lr=ADAM_LR); init={n:p.data.norm(2).item() for n,p in m.named_parameters()}
        rec=dict(seed=seed,accs=[],dead=[],erank=[],gns=[],wdrift=[],task1=None,dead_t1=None,healthy=False,tc=None)
        for t in range(N_TASKS):
            p=P[t]; train(m,o,X_tr[:,p],y_tr,crit); pr=probe(X_te[:,p],seed*10000+t)
            a=acc(m,X_te[:,p],y_te); d=dead(m,pr); er=erank(m,pr); g=gns(m,X_tr[:,p],y_tr,crit); w=wdrift(m,init)
            rec['accs'].append(a); rec['dead'].append(d); rec['erank'].append(er); rec['gns'].append(g); rec['wdrift'].append(w)
            if t==0: rec['task1']=a; rec['dead_t1']=d; rec['healthy']=(a>0.8 and d<0.25)
            rec['tc']=collapse(rec['accs'])
            if t<3 or (t+1)%50==0 or t==N_TASKS-1: print(f"  s{seed} t{t+1} acc={a:.4f} dead={d:.3f} erank={er:.2f} gns={g:.2f} [{time.time()-t0:.0f}s]",flush=True)
        recs[seed]=rec; trajs[f"seed_{seed}"]={k:rec[k] for k in ['accs','dead','erank','gns','wdrift']}
        json.dump(trajs,open(TRAJ,"w"))
        f20=np.mean(rec['accs'][:20]); l20=np.mean(rec['accs'][-20:])
        print(f"  Seed {seed} DONE dead_t1={rec['dead_t1']:.3f} drop={(f20-l20)*100:.1f}pp tc={rec['tc']}",flush=True)
    R=list(recs.values())
    healthy=all(r['healthy'] for r in R); ncol=sum(r['tc'] is not None for r in R)
    drop=float(np.mean([(np.mean(r['accs'][:20])-np.mean(r['accs'][-20:]))*100 for r in R]))
    auc={'dead_unit_fraction':pauc(R,'dead',False),'effective_rank':pauc(R,'erank',True),
         'gradient_noise_scale':pauc(R,'gns',False),'weight_norm_drift':pauc(R,'wdrift',False),
         'monotone_null_taskindex':pauc(R,None,False,taskidx=True)}
    leads={}
    for key in ['dead','erank','gns','wdrift']:
        ls=[]
        for r in R:
            if r['tc'] is None: continue
            on=onset(r[key]);
            if on is not None: ls.append(r['tc']-on)
        leads[key]=float(np.median(ls)) if ls else None
    out={"status":"DONE","optimizer":"Adam","lr":ADAM_LR,"regime":"hidden=100 MLP online Permuted-MNIST, ADAM",
         "n_seeds":N_SEEDS,"n_tasks":N_TASKS,"healthy":bool(healthy),"n_collapsed":int(ncol),
         "mean_dead_after_task1":float(np.mean([r['dead_t1'] for r in R])),"mean_task1_acc":float(np.mean([r['task1'] for r in R])),
         "mean_acc_drop_pp":drop,"plasticity_loss_confirmed":bool(drop>5),
         "predictive_auc":{k:(None if (v is None or not math.isfinite(v)) else round(v,4)) for k,v in auc.items()},
         "lead_median":leads,
         "subject_executed":f"Adam lr={ADAM_LR}, {N_SEEDS} seeds x {N_TASKS} tasks precedence + monotone-null",
         "metrics":{"optimizer":"Adam","healthy":bool(healthy),"plasticity_loss_confirmed":bool(drop>5),
                    "mean_acc_drop_pp":drop,"predictive_auc":{k:(None if (v is None or not math.isfinite(v)) else round(v,4)) for k,v in auc.items()}},
         "notes":"Does plasticity loss + the dead/erank>GNS precedence hold under Adam? Monotone-null included."}
    json.dump(out,open(RP,"w"),indent=2); print("DONE",json.dumps(out["metrics"],indent=2),flush=True)

if __name__=="__main__": main()
