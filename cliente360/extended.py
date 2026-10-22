"""Experimentos adicionales de la propuesta. Sin modificar la red principal.
La version por categoria agrega logits de producto (media por categoria).
La recompra se compara con clasificadores de referencia independientes.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from .settings import OUT,ART,REPORT
from .network import MultiTask
from .metrics import ranking

def run(source='synthetic'):
    d=dict(np.load(OUT/f'{source}.npz'));net=MultiTask.load(ART/f'{source}_multi.npz')
    n_users=len(d['x_train']);n_products=len(d['category_idx']);cat=d['category_idx'];ncat=int(cat.max())+1
    train=d['pairs_train'];counts=np.bincount(train[:,1],minlength=n_products)
    category_rows=[];sparse_rows=[];binary_rows=[]
    popcat=np.bincount(cat[train[:,1]],minlength=ncat).astype(float)
    popscore=np.tile(counts,(n_users,1))
    ytrain=(d['y_train']>0).astype(int)
    models={}
    if len(np.unique(ytrain))==2:
        models={'Regresión logística':LogisticRegression(max_iter=500,random_state=42),'Árboles potenciados':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=42)}
        for model in models.values():model.fit(d['x_train'],ytrain)
    for split in ['val','test']:
        pairs=d[f'pairs_{split}'];scores=net.scores(d[f'x_{split}'])
        catscores=np.column_stack([scores[:,cat==c].mean(axis=1) for c in range(ncat)])
        cp=np.unique(np.column_stack([pairs[:,0],cat[pairs[:,1]]]),axis=0)
        for name,s in [('Red conjunta · agregación por categoría',catscores),('Popularidad de categoría',np.tile(popcat,(n_users,1)))]:
            category_rows.append({'source':source,'split':split,'model':name,'categories':ncat,**ranking(s,cp,k=3)})
        for label,mask in [('0-2 interacciones train',counts<=2),('3-10 interacciones train',(counts>=3)&(counts<=10)),('11+ interacciones train',counts>10)]:
            relevant=pairs[mask[pairs[:,1]]]
            for name,s in [('Red conjunta',scores),('Popularidad',popscore)]:
                sparse_rows.append({'source':source,'split':split,'group':label,'products':int(mask.sum()),'relevant_pairs':len(relevant),'model':name,**ranking(s,relevant)})
        y=(d[f'y_{split}']>0).astype(int)
        predictions={'Tasa de recompra train':np.full(len(y),float(ytrain.mean()))}
        for name,model in models.items():predictions[name]=model.predict_proba(d[f'x_{split}'])[:,1]
        for name,p in predictions.items():
            binary_rows.append({'source':source,'split':split,'model':name,'customers':len(y),'positive_rate':float(y.mean()),'ROC_AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'PR_AUC_AP':float(average_precision_score(y,p)) if y.sum() else None,'Brier':float(brier_score_loss(y,p)),'LogLoss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]))})
    pd.DataFrame(category_rows).to_csv(REPORT/f'{source}_category_rec.csv',index=False)
    pd.DataFrame(sparse_rows).to_csv(REPORT/f'{source}_sparse_products.csv',index=False)
    pd.DataFrame(binary_rows).to_csv(REPORT/f'{source}_repurchase.csv',index=False)
    note={'category_method':'Media de logits de productos por categoría; sin entrenamiento independiente ni nueva cabeza. Popularidad por pares cliente-producto train. Evaluación de catálogo completo de categorías con K=3.','repurchase_method':'Target binario gasto futuro>0. Clasificadores independientes y tasa train. No es cabeza adicional de la red ni probabilidad causal de responder a publicidad.','sparse_products_method':'Estratos según número de pares cliente-producto train. Se restringen positivos de evaluación al estrato, manteniendo candidatos de catálogo completo. No mide productos completamente desconocidos.'}
    (REPORT/f'{source}_extended_notes.json').write_text(json.dumps(note,indent=2,ensure_ascii=False))
    return category_rows,sparse_rows,binary_rows
