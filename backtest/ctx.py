import numpy as np, pandas as pd, time
from chronos import BaseChronosPipeline
D=""
data=pd.read_csv(D+"train.csv",parse_dates=["ds"]); wf=pd.read_csv(D+"weather_forecast.csv",parse_dates=["ds"])
W=["cloud_cover","temperature","wind_speed","daylight","humidity"]; H=48
Q=[0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
pipe=BaseChronosPipeline.from_pretrained("amazon/chronos-2",device_map="cpu")
df=data.copy(); df["id"]="s"
N=len(data); idx=[N-H-24*j for j in range(600)][::-1]
for CTX in [1024,4096,8192]:
    out=[]; t0=time.time()
    for o in idx:
        q=pipe.predict_df(df.iloc[max(0,o-CTX):o][["id","ds","y"]+W],future_df=df.iloc[o:o+H][["id","ds"]+W],prediction_length=H,
            quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")[[str(x) for x in Q]].to_numpy()
        for h in range(H): out.append(dict(origin=data.ds.iloc[o-1],h=h+1,**{f"q{x}":q[h,k] for k,x in enumerate(Q)}))
    pd.DataFrame(out).to_parquet(f"bt600_ctx{CTX}.parquet")
    q=pipe.predict_df(df.iloc[-CTX:][["id","ds","y"]+W],future_df=wf.assign(id="s")[["id","ds"]+W],prediction_length=H,
            quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")[[str(x) for x in Q]]
    q.columns=[f"q{x}" for x in Q]; q.to_parquet(f"final_ctx{CTX}.parquet"); print(CTX,time.time()-t0,flush=True)
