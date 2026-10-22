"""Capa de consulta compartida por panel y API. Sin reentrenar al consultar."""
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from .settings import OUT,ART,REPORT,SYN,RAW
from .network import MultiTask

@lru_cache(maxsize=2)
def bundle(source='synthetic'):
    if source not in ['synthetic','original']:raise ValueError('Escenario no valido')
    return (dict(np.load(OUT/f'{source}.npz')),MultiTask.load(ART/f'{source}_multi.npz'),json.loads((OUT/'mapping.json').read_text()),pd.read_parquet(OUT/f'{source}_segments.parquet'),pd.read_parquet(OUT/'products.parquet'))

def summary(source='synthetic'):
    d,net,m,seg,prod=bundle(source)
    return {'source':source,'customers':len(m['users']),'products':len(prod),'cut':'2018-04-01','horizon_end':'2018-07-01','currency':'BRL','future_purchase_rate':float((d['y_test']>0).mean()),'audit':json.loads((REPORT/'synthetic_audit.json').read_text()),'backend':json.loads((ART/f'{source}_training.json').read_text())['backend']}

def customers(source='synthetic'):return bundle(source)[2]['users']

@lru_cache(maxsize=2)
def inference(source='synthetic'):
    d,net,_,_,_=bundle(source)
    return net.predict(d['x_test']),net.scores(d['x_test'])

def customer(customer_id,source='synthetic',k=10,exclude_seen=False):
    d,net,m,seg,prod=bundle(source)
    if customer_id not in m['users']:raise KeyError('Cliente fuera de la cohorte. No se inventa CLV para un ID desconocido.')
    u=m['users'].index(customer_id);predictions,all_scores=inference(source);scores=all_scores[u].copy()
    if exclude_seen:
        h=d['history_test'];scores[h[h[:,0]==u,1]]=-np.inf
    rank=np.argsort(-scores)[:min(k,len(prod))];records=[]
    for p in rank:
        if not np.isfinite(scores[p]):continue
        records.append({'product_id':m['products'][p],'category':str(prod.iloc[p].category),'affinity_score':float(scores[p])})
    if len(d['pairs_train'])==0: records=[]
    row=seg.iloc[u]
    return {'customer_id':customer_id,'source':source,'clv_3m_brl':float(predictions[u]),'segment':int(row.segment),'frequency':int(row.frequency),'recency_days':float(row.recency),'historical_spend_brl':float(row.monetary),'recommendations':records,'warning':('Sin interacciones de recomendacion en train para esta cohorte/catalogo: no se emiten recomendaciones.' if len(d['pairs_train'])==0 else 'Score de ranking no calibrado, no es probabilidad de compra. Ventana historica de demostracion, no prediccion actual.')}

def segments(source='synthetic'):
    return {'points':bundle(source)[3][['customer_unique_id','segment','pca_1','pca_2','clv_pred','frequency','recency']].to_dict('records'),'profiles':pd.read_csv(REPORT/f'{source}_profiles.csv').to_dict('records'),'metrics':pd.read_csv(REPORT/f'{source}_segmentation.csv').to_dict('records')}

def evaluation(source='synthetic'):
    clv=pd.concat([pd.read_csv(REPORT/f'{source}_{n}_clv.csv') for n in ['neural','baseline']])
    rec=pd.concat([pd.read_csv(REPORT/f'{source}_{n}_rec.csv') for n in ['neural','baseline']])
    sensitivity=REPORT/f'{source}_sensitivity.csv'
    return {'clv':clv.to_dict('records'),'recommendation':rec.to_dict('records'),'sensitivity':pd.read_csv(sensitivity).to_dict('records') if sensitivity.exists() else [],'campaign':json.loads((REPORT/f'{source}_campaign.json').read_text()),'sparse_history':pd.read_csv(REPORT/f'{source}_sparse_history.csv').to_dict('records')}

@lru_cache(maxsize=2)
def all_history(source='synthetic'):
    from .data import load_orders
    folder=SYN if source=='synthetic' else RAW
    o=load_orders(folder);o=o[o.order_purchase_timestamp<'2018-04-01'].copy()
    ids=set(pd.read_csv(SYN/'provenance.csv').order_id) if source=='synthetic' else set()
    o['is_synthetic']=o.order_id.isin(ids)
    return o

def history(customer_id,source='synthetic'):
    o=all_history(source);return o[o.customer_unique_id==customer_id][['order_id','order_purchase_timestamp','amount','is_synthetic']].assign(order_purchase_timestamp=lambda x:x.order_purchase_timestamp.astype(str)).sort_values('order_purchase_timestamp').to_dict('records')
