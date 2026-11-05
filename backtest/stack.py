import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
b=pd.read_parquet("bt600.parquet"); d=pd.read_csv("train.csv",parse_dates=["ds"])
W=["cloud_cover","temperature","wind_speed","humidity"]
b=b.merge(d[["ds"]+W],on="ds")
y=d.y.to_numpy(); res=y[24:]-y[:-24]; sig=np.sqrt(np.mean(res**2))
pos=pd.Series(np.arange(len(d)),index=d.ds)
o_i=pos[b.origin].to_numpy(); hh=b.h.to_numpy()
b["sn"]=y[o_i-24+((hh-1)%24)+1-0]
k=(hh-1)//24; b["sn_l"]=(b.sn-1.96*sig*np.sqrt(k+1)).clip(0); b["sn_u"]=b.sn+1.96*sig*np.sqrt(k+1)
b["hour"]=b.ds.dt.hour; b["doy"]=b.ds.dt.dayofyear
b["lvl"]=[y[i-336:i].mean() for i in o_i]
QC=[c for c in b.columns if c.startswith("q0")]
for c in QC: b[c+"_r"]=b[c]/b.lvl
F=QC+[c+"_r" for c in QC]+["h","hour","daylight","lvl","sn"]+W
def wink(y,l,u): return (u-l)+40*np.clip(l-y,0,None)+40*np.clip(y-u,0,None)
origins=np.sort(b.origin.unique())
test_o=origins[-150:]
out=[]
for s in range(0,150,30):
    tr=b[b.origin<test_o[s]-pd.Timedelta(hours=48)]; te=b[b.origin.isin(test_o[s:s+30])].copy()
    for a,nm in [(0.025,"st_l"),(0.975,"st_u")]:
        m=HGB(loss="quantile",quantile=a,max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50).fit(tr[F],tr.y)
        te[nm]=m.predict(te[F])
    out.append(te)
t=pd.concat(out); t["st_l"]=t.st_l.clip(0); t["st_u"]=np.maximum(t.st_u,t.st_l)
t["half"]=np.where(t.h<=24,"pub","priv")
for nm,l,u in [("chronos",t["q0.025"].clip(0),t["q0.975"]),("snaive",t.sn_l,t.sn_u),("stack",t.st_l,t.st_u),
               ("avg",(t["q0.025"].clip(0)+t.st_l)/2,(t["q0.975"]+t.st_u)/2)]:
    g=t.assign(w=wink(t.y,l,u),c=(t.y>=l)&(t.y<=u),wd=u-l).groupby("half")[["w","c","wd"]].mean().round(3)
    print(nm, g.to_dict("index"))
t.to_parquet("stack_oos.parquet")
