from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
from torch import nn
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'code'))
import experiment as e
out=root/'results/relative_price/diagnostics'
out.mkdir(parents=True,exist_ok=True)
world=e.experimental_worlds()['higher_elasticity']; seed=0
train,test=e.split_and_add_item_mean(e.generate_panel(world,seed))
features=e.PREDICTIVE_FEATURES
results=[]; losses=[]
def diagnose(model, data, label,epoch):
 logs=e.predictor_log_demand_at_candidate_prices(model,data)
 ms=e.PRICE_MULTIPLIERS; prices=data.price.to_numpy()[:,None]*ms
 costs=data.unit_cost.to_numpy()[:,None]
 pred=np.exp(logs); true=e.simulator_demand_at_candidate_prices(test,world,np.log(ms))
 fp=(prices-costs)*pred; actions=fp.argmax(1); tp=(prices-costs)*true
 regret=100*(1-tp[np.arange(len(data)),actions].sum()/tp.max(1).sum())
 slope=(logs[:,-1]-logs[:,0])/(np.log(ms[-1])-np.log(ms[0]))
 mse=float(np.mean((model.predict(data[features].to_numpy())-test.log_units.to_numpy())**2))
 result={'variant':label,'epoch':epoch,'regret_pct':float(regret),'upper_share':float(np.mean(actions==6)), 'implied_log_price_slope_quantiles':np.quantile(slope,[0,.1,.5,.9,1]).tolist(),'median_q_up15_vs_current':float(np.median(np.exp(logs[:,6]-logs[:,3]))),'test_log_mse':mse,'monotone_profit_share':float(np.mean(np.all(np.diff(fp,axis=1)>0,axis=1)))}
 results.append(result);print(json.dumps(result),flush=True)
 for row in [0,100,1000]:
  for k,m in enumerate(ms):
   curves.append({'variant':label,'epoch':epoch,'row':row,'multiplier':float(m),'predicted_demand':float(pred[row,k]),'predicted_profit':float(fp[row,k]),'true_demand':float(true[row,k]),'true_profit':float(tp[row,k])})
 (out/'transformer_probe_results.json').write_text(json.dumps({'results':results,'losses':losses,'curves':curves},indent=2))
curves=[]
for label,maxepoch in [('original_features',36),('relative_price_diagnostic',12)]:
 tr=train.copy();te=test.copy()
 if label!='original_features':
  means=train.groupby('item_id').log_price.mean()
  tr.log_price=tr.log_price-tr.item_id.map(means)
  te.log_price=te.log_price-te.item_id.map(means)
 model=e.TransformerRegressor(seed,width=32,n_layers=2)
 x=tr[features].to_numpy(dtype=float);y=tr.log_units.to_numpy(dtype=float)
 model.x_mean=x.mean(0);model.x_std=np.maximum(x.std(0),1e-6);model.y_mean=float(y.mean());model.y_std=max(float(y.std()),1e-6)
 xt=torch.from_numpy(((x-model.x_mean)/model.x_std).astype(np.float32)).to(model.device)
 yt=torch.from_numpy(((y-model.y_mean)/model.y_std).astype(np.float32)).to(model.device)
 opt=torch.optim.AdamW(model.network.parameters(),lr=1e-3,weight_decay=1e-4)
 start=time.time()
 for epoch in range(maxepoch):
  model.network.train();total=0.
  gen=torch.Generator().manual_seed(10000+epoch)
  for idx in torch.randperm(len(xt),generator=gen).split(2048):
   idx=idx.to(model.device);opt.zero_grad(set_to_none=True)
   loss=nn.functional.mse_loss(model.network(xt[idx]),yt[idx]);loss.backward();opt.step()
   total+=loss.item()*len(idx)
  losses.append({'variant':label,'epoch':epoch+1,'online_standardized_train_mse':total/len(xt)})
  if (epoch+1)%6==0:print(label,'epoch',epoch+1,'train_mse',total/len(xt),'seconds',round(time.time()-start),flush=True)
  if epoch+1 in [12,36]:diagnose(model,te,label,epoch+1)
print('DONE',flush=True)
