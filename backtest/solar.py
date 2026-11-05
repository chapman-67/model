import numpy as np, pandas as pd, time
from chronos import BaseChronosPipeline
D=""
data=pd.read_csv(D+"train.csv",parse_dates=["ds"]); wf=pd.read_csv(D+"weather_forecast.csv",parse_dates=["ds"])
def add(f):
    f=f.copy(); doy=f.ds.dt.dayofyear.values; hr=f.ds.dt.hour.values-0.5
    lat=np.radians(50.0); dec=np.radians(23.44)*np.sin(2*np.pi*(284+doy)/365)
    ha=np.radians(15*(hr-12.5))
    el=np.clip(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha),0,None)
    f["sun"]=el; f["clear_sky"]=el*(1-0.75*f.cloud_cover**3.4)
    return f
data=add(data); wf=add(wf)
Q=[0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]; H=48; CTX=2048
pipe=BaseChronosPipeline.from_pretrained("amazon/chronos-2",device_map="cpu")
df=data.copy(); df["id"]="s"; N=len(data); idx=[N-H-24*j for j in range(600)][::-1]
for name,W in [("solar",["cloud_cover","temperature","wind_speed","daylight","humidity","sun","clear_sky"])]:
    out=[]
    for o in idx:
        q=pipe.predict_df(df.iloc[o-CTX:o][["id","ds","y"]+W],future_df=df.iloc[o:o+H][["id","ds"]+W],prediction_length=H,
            quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")[[str(x) for x in Q]].to_numpy()
        for h in range(H): out.append(dict(origin=data.ds.iloc[o-1],h=h+1,**{f"q{x}":q[h,k] for k,x in enumerate(Q)}))
    pd.DataFrame(out).to_parquet(f"bt600_{name}.parquet")
    q=pipe.predict_df(df.iloc[-CTX:][["id","ds","y"]+W],future_df=wf.assign(id="s")[["id","ds"]+W],prediction_length=H,
            quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")[[str(x) for x in Q]]
    q.columns=[f"q{x}" for x in Q]; q.to_parquet(f"final_{name}.parquet")
print("sun>0 vs daylight agreement:",((data.sun>0)==(data.daylight==1)).mean().round(3))
