import numpy as np, pandas as pd
A=pd.read_parquet("bt600.parquet").sort_values(["origin","h"]).reset_index(drop=True)
B=pd.read_parquet("bt600_nowx.parquet").sort_values(["origin","h"]).reset_index(drop=True)
C=pd.read_parquet("bt600_log.parquet").sort_values(["origin","h"]).reset_index(drop=True)
assert (A.origin==B.origin).all() and (A.origin==C.origin).all()
QC=[c for c in A.columns if c.startswith("q0")]
y=A.y.to_numpy(); priv=A.h>24; rec=A.origin>="2026-05-01"
def wink(l,u): return (u-l)+40*np.clip(l-y,0,None)+40*np.clip(y-u,0,None)
def rule(M): 
    med=M["q0.5"].to_numpy(); return np.clip(np.where(med<200,M["q0.3"],med),0.5,None)
def ev(name,M,s=1.0):
    p=rule(M); med=M["q0.5"].to_numpy()
    l=np.clip(med-s*(med-M["q0.025"].to_numpy()),0,None); u=med+s*(M["q0.975"].to_numpy()-med)
    r={"m":name}
    for tag,m in [("all",priv),("rec",priv&rec)]:
        r[tag+"_RMSE"]=np.sqrt(((y-p)**2)[m].mean()); r[tag+"_MAPE"]=(100*abs(y-p)/y)[m].mean()
        r[tag+"_W"]=wink(l,u)[m].mean(); r[tag+"_cov"]=((y>=l)&(y<=u))[m].mean()
    return r
avg=lambda *Ms: pd.DataFrame({c:np.mean([M[c].to_numpy() for M in Ms],0) for c in QC})
R=[ev("A wx",A),ev("B nowx",B),ev("C log",C),ev("A+B",avg(A,B)),ev("A+C",avg(A,C)),ev("A+B+C",avg(A,B,C)),
   ev("A+C s0.9",avg(A,C),0.9),ev("A s0.7",A,0.7),ev("A+C s0.7",avg(A,C),0.7)]
print(pd.DataFrame(R).round(3).to_string())
