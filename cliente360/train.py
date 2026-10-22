import copy,json
import numpy as np
import pandas as pd
from .settings import OUT,ART,REPORT,SEED
from .network import MultiTask,sample_negatives
from .metrics import regression,ranking

def train(data,mode='multi',dim=16,lam=1.,negatives=3,decay=1e-4,epochs=160,seed=SEED,backend='numpy',transform='raw'):
    model=MultiTask(len(data['x_train']),len(data['category_idx']),data['x_train'].shape[1],data['category_idx'],data['product_attributes'],dim,seed)
    model.transform=transform
    ylog=np.log1p(data['y_train']) if transform=='log' else data['y_train'];model.ymean=float(ylog.mean());model.ystd=float(max(ylog.std(),.1));y=(ylog-model.ymean)/model.ystd
    pairs,labels=sample_negatives(data['pairs_train'],data['history_train'],len(data['category_idx']),negatives,seed)
    vp,vl=sample_negatives(data['pairs_val'],data['history_val'],len(data['category_idx']),negatives,seed+1)
    cw=0. if mode=='rec_only' else 1.;rw=0. if mode=='clv_only' else lam
    best=float('inf');patience=0;bestp=None;curve=[]
    if backend=='torch':
        from .torch_backend import TorchTrainer
        trainer=TorchTrainer(model,lr=.01,decay=decay)
    for epoch in range(epochs):
        if backend=='numpy':cl,rl=model.update(data['x_train'],y,pairs,labels,rw,cw,decay=decay)
        else:cl,rl=trainer.update(data['x_train'],y,pairs,labels,rw,cw);trainer.export(model)
        z=model.encode(data['x_val']);pred=z@model.p['w']+model.p['cb'][0]
        target=((np.log1p(data['y_val']) if transform=='log' else data['y_val'])-model.ymean)/model.ystd
        logits=model.scores(data['x_val'])[vp[:,0],vp[:,1]]
        valcl=float(np.mean((pred-target)**2));valrec=float(np.mean(np.logaddexp(0,logits)-vl*logits)) if len(vl) else 0.;loss=cw*valcl+rw*valrec
        curve.append({'epoch':epoch+1,'clv_loss':cl,'rec_loss':rl,'val_loss':loss,'val_clv_loss':valcl,'val_rec_loss':valrec})
        if loss<best-1e-5:best=loss;bestp=copy.deepcopy(model.p);patience=0
        else:patience+=1
        if patience>=20:break
    model.p=bestp
    # Correccion de retransf. estimada SOLO con residuos train.
    residual=ylog-((model.encode(data['x_train'])@model.p['w']+model.p['cb'][0])*model.ystd+model.ymean)
    model.smear=float(np.exp(np.clip(residual,-10,10)).mean()) if transform=='log' else 1.
    return model,curve

def run(source='synthetic',backend='numpy',epochs=160):
    data=dict(np.load(OUT/f'{source}.npz'));rows=[];recs=[]
    for mode in ['multi','clv_only','rec_only']:
        print(f'Entrenando {source} / {mode} / {backend}',flush=True)
        model,curve=train(data,mode=mode,lam=2. if mode=="multi" else 1.,backend=backend,epochs=epochs)
        model.save(ART/f'{source}_{mode}.npz')
        pd.DataFrame(curve).to_csv(REPORT/f'{source}_{mode}_learning.csv',index=False)
        for split in ['val','test']:
            if mode!='rec_only':rows.append({'source':source,'model':mode,'split':split,**regression(data[f'y_{split}'],model.predict(data[f'x_{split}']))})
            if mode!='clv_only':recs.append({'source':source,'model':mode,'split':split,**ranking(model.scores(data[f'x_{split}']),data[f'pairs_{split}'])})
    pd.DataFrame(rows).to_csv(REPORT/f'{source}_neural_clv.csv',index=False)
    pd.DataFrame(recs).to_csv(REPORT/f'{source}_neural_rec.csv',index=False)
    (ART/f'{source}_training.json').write_text(json.dumps({'backend':backend,'device':('cuda' if __import__('torch').cuda.is_available() and __import__('os').environ.get('CLIENTE360_DEVICE','auto')!='cpu' else 'cpu') if backend=='torch' else 'cpu','seed':SEED,'epochs_limit':epochs,'dim':16,'lambda':2.,'negatives':3,'target':'gasto_3_meses','smearing':'residuos train','note':'No se presupone superioridad de la red.'},indent=2))
    return rows,recs
