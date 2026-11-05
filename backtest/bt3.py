import sys, numpy as np, pandas as pd, time
from chronos import BaseChronosPipeline
D=""
data=pd.read_csv(D+"train.csv",parse_dates=["ds"]); wf=pd.read_csv(D+"weather_forecast.csv",parse_dates=["ds"])
W=["cloud_cover","temperature","wind_speed","daylight","humidity"]; H=48; CTX=2048
Q=[0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
pipe=BaseChronosPipeline.from_pretrained("amazon/chronos-2",device_map="cpu")
df=data.copy(); df["id"]="s"; df["ly"]=np.log1p(df.y)
def fc(past,fut,variant):
    if variant=="nowx":
        o=pipe.predict_df(past.iloc[-CTX:][["id","ds","y"]],prediction_length=H,quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")
        return o[[str(q) for q in Q]].to_numpy()
    o=pipe.predict_df(past.iloc[-CTX:][["id","ds","ly"]+W],future_df=fut.assign(id="s")[["id","ds"]+W],
        prediction_length=H,quantile_levels=Q,id_column="id",timestamp_column="ds",target="ly")
    return np.expm1(o[[str(q) for q in Q]].to_numpy())
N=len(data); idx=[N-H-24*j for j in range(600)][::-1]
for v in ["nowx","log"]:
    out=[]; t0=time.time()
    for o in idx:
        q=fc(df.iloc[:o],df.iloc[o:o+H],v)
        for h in range(H): out.append(dict(origin=data.ds.iloc[o-1],h=h+1,**{f"q{q_}":q[h,k] for k,q_ in enumerate(Q)}))
    pd.DataFrame(out).to_parquet(f"bt600_{v}.parquet")
    pd.DataFrame(fc(df,wf.assign(ly=0),v),columns=[f"q{q}" for q in Q]).to_parquet(f"final_{v}.parquet"); print(v,time.time()-t0,flush=True)
