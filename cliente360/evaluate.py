import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from .settings import OUT,REPORT,ART
from .metrics import regression,ranking
from .network import MultiTask

def baselines(source='synthetic'):
    d=dict(np.load(OUT/f'{source}.npz'));x=d['x_train'];y=d['y_train'];rows=[];recs=[]
    models={'Media':DummyRegressor(),'Regresion lineal':Ridge(alpha=10),'Random Forest':RandomForestRegressor(n_estimators=100,min_samples_leaf=12,max_depth=10,random_state=42,n_jobs=2),'Gradient Boosting':HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=10,random_state=42)}
    for name,model in models.items():
        model.fit(x,y)
        for split in ['val','test']:rows.append({'source':source,'model':name,'split':split,**regression(d[f'y_{split}'],np.maximum(0,model.predict(d[f'x_{split}'])))})
    # Baselines de recomendacion: mismas etiquetas train, mismo catalogo y universo.
    pairs=d['pairs_train'];nu=len(x);np_=len(d['category_idx'])
    matrix=csr_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(nu,np_))
    popularity=np.asarray(matrix.sum(0)).ravel()
    svd=TruncatedSVD(n_components=min(16,np_-1),random_state=42);scores=svd.fit_transform(matrix)@svd.components_
    for split in ['val','test']:
        for name,s in [('Popularidad',np.tile(popularity,(nu,1))),('SVD',scores)]:recs.append({'source':source,'model':name,'split':split,**ranking(s,d[f'pairs_{split}'])})
    pd.DataFrame(rows).to_csv(REPORT/f'{source}_baseline_clv.csv',index=False)
    pd.DataFrame(recs).to_csv(REPORT/f'{source}_baseline_rec.csv',index=False)
    # Analisis descriptivo, NO efecto causal de campana ni ROI.
    net=MultiTask.load(ART/f'{source}_multi.npz');pred=net.predict(d['x_test']);truth=d['y_test'];n=max(1,int(np.ceil(nu*.1)))
    rng=np.random.default_rng(42);random_means=[truth[rng.choice(nu,n,replace=False)].mean() for _ in range(500)]
    result={'source':source,'n_targeted':n,'selected_observed_revenue':float(truth[np.argsort(-pred)[:n]].sum()),'random_expected_revenue':float(np.mean(random_means)*n),'random_mean_ci_low':float(np.quantile(random_means,.025)),'random_mean_ci_high':float(np.quantile(random_means,.975)),'warning':'Ranking retrospectivo. No estima ingresos incrementales ni efecto causal de publicidad.'}
    import json
    (REPORT/f'{source}_campaign.json').write_text(json.dumps(result,indent=2))
    # Pocos datos: estratos por historial observado, no verdadero cold-start.
    snap=pd.read_parquet(OUT/f'{source}_snapshots.parquet');s=snap[snap.split=='test'].reset_index(drop=True)
    cold=[]
    for label,mask in [('1-3 pedidos',s.frequency<=3),('4+ pedidos',s.frequency>3)]:
        if mask.any():cold.append({'group':label,'n':int(mask.sum()),**regression(truth[mask],pred[mask])})
    pd.DataFrame(cold).to_csv(REPORT/f'{source}_sparse_history.csv',index=False)
    return rows,recs
