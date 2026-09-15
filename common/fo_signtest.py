import json,sys,glob,os
from math import comb
def first(d):
    ev=[e for e in d.get("events",[]) if e.get("type")=="response"]
    return min(e["start_time"] for e in ev) if ev else None
def load(root):
    out={}
    for f in glob.glob(os.path.join(root,"**","events.json"),recursive=True):
        k=os.path.relpath(os.path.dirname(f),root)
        out[k]=first(json.load(open(f)))
    return out
def sign_p(a,b):
    n=a+b; k=min(a,b)
    return sum(comb(n,i) for i in range(0,k+1))*2/2**n if n else 1.0
A,B=sys.argv[1],sys.argv[2]
a,b=load(A),load(B)
ks=sorted(set(a)&set(b))
earlier=later=same=0
for k in ks:
    x,y=a[k],b[k]
    if x is None or y is None: continue
    if abs(y-x)<1e-6: same+=1
    elif y<x: earlier+=1
    else: later+=1
print(f"{os.path.basename(B)} vs {os.path.basename(A)}: n={len(ks)} earlier={earlier} later={later} same={same} p={sign_p(earlier,later):.2g}")
