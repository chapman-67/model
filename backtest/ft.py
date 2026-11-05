import sys, numpy as np, pandas as pd, time, torch
from chronos import BaseChronosPipeline
from chronos.chronos2.preprocess import from_data_frame
D=""
data=pd.read_csv(D+"train.csv",parse_dates=["ds"]); wf=pd.read_csv(D+"weather_forecast.csv",parse_dates=["ds"])
W=["cloud_cover","temperature","wind_speed","daylight","humidity"]; H=48; CTX=2048
Q=[0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
mode=sys.argv[1]; steps=int(sys.argv[2]); lr=float(sys.argv[3]); final=len(sys.argv)>4
df=data.copy(); df["id"]="s"; N=len(data)
idx=[N-H-24*j for j in range(180)][::-1]
cut=N if final else idx[0]-H
base=BaseChronosPipeline.from_pretrained("amazon/chronos-2",device_map="cpu")
tr=from_data_frame(df.iloc[:cut][["id","ds","y"]+W],target_columns=["y"],prediction_length=H,known_covariates_names=W,id_column="id",timestamp_column="ds")
t0=time.time()
pipe=base.fit(tr,prediction_length=H,finetune_mode=mode,learning_rate=lr,num_steps=steps,batch_size=16,context_length=CTX,
              output_dir=f"ft_{mode}_{steps}_{lr}{'_final' if final else ''}",remove_printer_callback=True)
print("fit s",round(time.time()-t0),flush=True)
def fc(past,fut):
    o=pipe.predict_df(past.iloc[-CTX:][["id","ds","y"]+W],future_df=fut.assign(id="s")[["id","ds"]+W],prediction_length=H,
        quantile_levels=Q,id_column="id",timestamp_column="ds",target="y")
    return o[[str(q) for q in Q]].to_numpy()
tag=f"{mode}_{steps}_{lr}"
if final:
    pd.DataFrame(fc(df,wf),columns=[f"q{q}" for q in Q]).to_parquet(f"final_ft_{tag}.parquet")
else:
    out=[]
    for o in idx:
        q=fc(df.iloc[:o],df.iloc[o:o+H])
        for h in range(H): out.append(dict(origin=data.ds.iloc[o-1],h=h+1,**{f"q{x}":q[h,k] for k,x in enumerate(Q)}))
    pd.DataFrame(out).to_parquet(f"bt180_ft_{tag}.parquet")
print("done",tag,round(time.time()-t0),flush=True)
