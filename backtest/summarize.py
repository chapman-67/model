import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; B=sys.argv[1] if len(sys.argv)>1 else "backtest"; OUT=sys.argv[2] if len(sys.argv)>2 else "results"
d=pd.read_csv("train.csv",parse_dates=["ds"]); y_all=d.y.to_numpy()
A=pd.read_parquet(f"{B}/bt600.parquet").sort_values(["origin","h"]).reset_index(drop=True)
S=pd.read_parquet(f"{B}/bt600_solar.parquet").sort_values(["origin","h"]).reset_index(drop=True)
T=pd.read_parquet(f"{B}/bt600_tf3.parquet").sort_values(["origin","h"]).reset_index(drop=True)
Bn=pd.read_parquet(f"{B}/bt600_nowx.parquet").sort_values(["origin","h"]).reset_index(drop=True)
L=pd.read_parquet(f"{B}/bt600_log.parquet").sort_values(["origin","h"]).reset_index(drop=True)
y=A.y.to_numpy()
wind=A.merge(d[["ds","wind_speed"]],on="ds",how="left").wind_speed.to_numpy(); night=(A.daylight==0).to_numpy()
pos=pd.Series(np.arange(len(d)),index=d.ds); oi=pos[A.origin].to_numpy()+1; hh=A.h.to_numpy()
sn=y_all[oi-24+(hh-1)%24]; sig=np.array([np.std(y_all[24:i]-y_all[:i-24],ddof=1) for i in oi]) if False else np.std(y_all[24:]-y_all[:-24],ddof=1)
half=1.96*sig*np.sqrt((hh-1)//24+1); snl=np.clip(sn-half,0,None); snu=sn+half
rule=lambda med,low: np.clip(np.where(med<200,low,med),0.5,None)
def interval(Q,s):
    med=Q["q0.5"].to_numpy(); return np.clip(med-s*(med-Q["q0.025"].to_numpy()),0,None), med+s*(Q["q0.975"].to_numpy()-med)
def metr(p,l,u,m):
    w=(u-l)+40*np.clip(l-y,0,None)+40*np.clip(y-u,0,None)
    return dict(RMSE=np.sqrt(((y-p)**2)[m].mean()),MAPE=(100*abs(y-p)/y)[m].mean(),Winkler=w[m].mean(),coverage=((y>=l)&(y<=u))[m].mean())
models={"seasonal naive (benchmark)":(sn,snl,snu),
 "Chronos-2, no weather":(rule(Bn["q0.5"],Bn["q0.3"]),*interval(Bn,1)),
 "Chronos-2, log target":(rule(L["q0.5"],L["q0.3"]),*interval(L,1)),
 "Chronos-2 + weather (Session 1)":(rule(A["q0.5"],A["q0.3"]),*interval(A,1)),
 "Chronos-2 + weather + solar":(rule(S["q0.5"],S["q0.3"]),*interval(S,1)),
 "TimesFM 3 + weather + solar":(rule(T.tf,T.tq2),*interval(S,1)),
 "mean(Chronos, TimesFM), interval x0.6":(rule((S["q0.5"]+T.tf)/2,(S["q0.3"]+T.tq2)/2),*interval(S,0.6))}
medM=((S["q0.5"]+T.tf)/2).to_numpy(); pF=rule(medM,((S["q0.3"]+T.tq2)/2).to_numpy())
pF=np.where(night&(wind<=3)&(medM<300),np.clip(((S["q0.1"]+T.tq0)/2).to_numpy(),0.5,None),pF)
lF,uF=interval(S,0.7); models["FINAL: mean + calm-night q0.1, interval x0.7"]=(pF,np.minimum(lF,pF),np.maximum(uF,pF))
rows=[]
for n,(p,l,u) in models.items():
    for part,m in [("hours 1-24",A.h<=24),("hours 25-48",A.h>24)]:
        rows.append(dict(model=n,part=part,**metr(p,l,u,m.to_numpy())))
tab=pd.DataFrame(rows).round(3); tab.to_csv(f"{OUT}/backtest_600_windows.csv",index=False); print(tab.to_string())
r=[]
for s in [0.6,0.7,0.8,0.9,1.0,1.1]:
    l,u=interval(A,s); w=(u-l)+40*np.clip(l-y,0,None)+40*np.clip(y-u,0,None)
    r.append(pd.DataFrame({"month":A.origin.dt.to_period("M").astype(str),"s":s,"w":w,"cov":(y>=l)&(y<=u)}))
cal=pd.concat(r).groupby(["month","s"]).agg(W=("w","mean"),cov=("cov","mean")).reset_index(); cal.to_csv(f"{OUT}/interval_scale_by_month.csv",index=False)
fig,ax=plt.subplots(figsize=(10,3.6))
h=d.iloc[-7*24:]; ax.plot(h.ds,h.y,color="black",lw=1,label="історія")
f=pd.read_csv("MASTYTSKYI_project.csv",parse_dates=["ds"]); s1=pd.read_csv(f"{OUT}/session1_submission.csv",parse_dates=["ds"])
ax.plot(f.ds,f.forecast,color="tab:blue",label="фінальний прогноз"); ax.fill_between(f.ds,f.lower_95,f.upper_95,color="tab:blue",alpha=.2,label="фінальний 95% інтервал")
ax.plot(s1.ds,s1.upper_95,color="tab:orange",ls=":",lw=1,label="верхня межа Session 1")
ax.legend(fontsize=8,loc="upper left"); ax.set_ylabel("генерація"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_forecast.png",dpi=150)
fig,ax=plt.subplots(figsize=(6,3.4)); g=cal.groupby("s").W.mean(); ax.plot(g.index,g.values,marker="o")
ax.set_xlabel("множник ширини (1 = Chronos 2.5/97.5%)"); ax.set_ylabel("середній Winkler, 600 днів"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_scale.png",dpi=150)
p,l,u=models["FINAL: mean + calm-night q0.1, interval x0.7"]; res=y-p
fig,ax=plt.subplots(1,2,figsize=(10,3.2))
pd.Series(res).groupby(A.ds.dt.hour.values).mean().plot.bar(ax=ax[0],color="tab:gray"); ax[0].set_title("середній залишок за годиною доби"); ax[0].set_xlabel("година")
pd.Series(res).groupby(A.h.values).apply(lambda s: np.sqrt((s**2).mean())).plot(ax=ax[1]); ax[1].set_title("RMSE за горизонтом (годин уперед)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_residuals.png",dpi=150)
pd.DataFrame({"hour":range(24),"mean_resid":pd.Series(res).groupby(A.ds.dt.hour.values).mean().values}).round(1).to_csv(f"{OUT}/residual_by_hour.csv",index=False)
print("bias overall",res.mean().round(2))
