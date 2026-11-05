import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
b=pd.read_parquet("bt600.parquet"); d=pd.read_csv("train.csv",parse_dates=["ds"])
W=["cloud_cover","temperature","wind_speed","humidity"]; b=b.merge(d[["ds"]+W],on="ds")
y=d.y.to_numpy(); pos=pd.Series(np.arange(len(d)),index=d.ds); oi=pos[b.origin].to_numpy()
b["hour"]=b.ds.dt.hour; b["last"]=y[oi]; b["lvl"]=[y[i-336:i].mean() for i in oi]; b["night_lvl"]=[np.median(y[i-24:i][d.daylight.values[i-24:i]==0]) for i in oi]
QC=[c for c in b.columns if c.startswith("q0")]
F=QC+["h","hour","daylight","last","lvl","night_lvl"]+W
origins=np.sort(b.origin.unique()); test=origins[-180:]
res=[]
for s in range(0,180,30):
    tr=b[b.origin<test[s]-pd.Timedelta(hours=48)]; te=b[b.origin.isin(test[s:s+30])].copy()
    for tgt in ["resid"]:
        m=HGB(max_iter=300,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=1.0).fit(tr[F],tr.y-tr["q0.5"])
        te["corr"]=m.predict(te[F])
    res.append(te)
t=pd.concat(res)
def rule(med,q3): return np.clip(np.where(med<200,q3,med),0.5,None)
def rep(name,p):
    for half,m in [("d1",t.h<=24),("d2",t.h>24)]:
        print(f"  {name:22s}{half} RMSE {np.sqrt(((t.y-p)**2)[m].mean()):.1f} MAPE {(100*abs(t.y-p)/t.y)[m].mean():.1f}")
rep("chronos (current)",rule(t["q0.5"],t["q0.3"]))
for a in [0.5,1.0]:
    sh=a*t["corr"]; rep(f"+{a}*correction",rule(t["q0.5"]+sh,t["q0.3"]+sh))
