import numpy as np, pandas as pd, time
from chronos import BaseChronosPipeline
D=""
data=pd.read_csv(D+"train.csv",parse_dates=["ds"])
W=["cloud_cover","temperature","wind_speed","daylight","humidity"]; H=48; CTX=2048
Q=[0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
pipe=BaseChronosPipeline.from_pretrained("amazon/chronos-2",device_map="cpu")
df=data.copy(); df["id"]="s"; rng=np.random.default_rng(0)
def noisy(f, lvl):
    f=f.copy(); day2=(np.arange(H)>=24)
    sc=np.where(day2,2*lvl,lvl)
    f["cloud_cover"]=np.clip(np.round((f.cloud_cover+rng.normal(0,0.25*sc,H))*4)/4,0,1)
    f["wind_speed"]=np.clip(np.round(f.wind_speed+rng.normal(0,1.0*sc,H)),0,None)
    f["temperature"]=np.round(f.temperature+rng.normal(0,1.5*sc,H))
    f["humidity"]=np.clip(np.round(f.humidity+rng.normal(0,5*sc,H)),0,100)
    return f
N=len(data); idx=[N-H-24*j for j in range(180)][::-1]
for lvl in [0.5,1.0]:
    out=[]
    for o in idx:
        fut=noisy(df.iloc[o:o+H],lvl)
        q=pipe.predict_df(df.iloc[o-CTX:o][["id","ds","y"]+W],future_df=fut[["id","ds"]+W],prediction_length=H,
            quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")[[str(x) for x in Q]].to_numpy()
        for h in range(H): out.append(dict(origin=data.ds.iloc[o-1],h=h+1,**{f"q{x}":q[h,k] for k,x in enumerate(Q)}))
    pd.DataFrame(out).to_parquet(f"bt180_noise{lvl}.parquet"); print(lvl,flush=True)
