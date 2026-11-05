import numpy as np, pandas as pd, time, timesfm
D=""
data=pd.read_csv(D+"train.csv",parse_dates=["ds"]); wf=pd.read_csv(D+"weather_forecast.csv",parse_dates=["ds"])
def add(f):
    f=f.copy(); doy=f.ds.dt.dayofyear.values; hr=f.ds.dt.hour.values-0.5
    lat=np.radians(50.0); dec=np.radians(23.44)*np.sin(2*np.pi*(284+doy)/365); ha=np.radians(15*(hr-12.5))
    el=np.clip(np.sin(lat)*np.sin(dec)+np.cos(lat)*np.cos(dec)*np.cos(ha),0,None)
    f["sun"]=el; f["clear_sky"]=el*(1-0.75*f.cloud_cover**3.4); return f
data=add(data); wf=add(wf)
W=["cloud_cover","temperature","wind_speed","daylight","humidity","sun","clear_sky"]; H=48; CTX=2048
m=timesfm.TimesFM3Forecaster.from_pretrained(device="cpu"); print("loaded",flush=True)
y=data.y.values.astype(float); X=data[W].values.astype(float)
N=len(data); idx=[N-H-24*j for j in range(600)][::-1]; out=[]; t0=time.time()
for i,o in enumerate(idx):
    r=m.predict(y[o-CTX:o],H,past_future_covariates=X[o-CTX:o+H].T,return_quantiles=True,make_positive=True)
    for h in range(H): out.append(dict(origin=data.ds.iloc[o-1],h=h+1,tf=r.forecast[h],**{f"tq{k}":r.quantiles[h,k] for k in range(r.quantiles.shape[1])}))
    if i%100==0: print(i,round(time.time()-t0),flush=True)
pd.DataFrame(out).to_parquet("bt600_tf3.parquet")
Xf=np.vstack([X[-CTX:],wf[W].values.astype(float)])
r=m.predict(y[-CTX:],H,past_future_covariates=Xf.T,return_quantiles=True,make_positive=True)
pd.DataFrame({"tf":r.forecast,**{f"tq{k}":r.quantiles[:,k] for k in range(r.quantiles.shape[1])}}).to_parquet("final_tf3.parquet"); print("done",flush=True)
