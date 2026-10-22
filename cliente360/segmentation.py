import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,adjusted_rand_score
from .settings import OUT,ART,REPORT
from .network import MultiTask

def segment(source='synthetic'):
    d=np.load(OUT/f'{source}.npz');net=MultiTask.load(ART/f'{source}_multi.npz')
    # K y centroides ajustados en train; no se seleccionan con test.
    spaces={'embeddings':(net.encode(d['x_train']),net.encode(d['x_test'])),'RFM':(d['x_train'][:,:3],d['x_test'][:,:3])}
    rows=[];assignments=None
    for name,(train,test) in spaces.items():
        scaler=StandardScaler().fit(train);a=scaler.transform(train);b=scaler.transform(test);best=None;bestscore=-1
        for k in range(2,7):
            km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(a)
            sil=float(silhouette_score(a,km.labels_,sample_size=min(1000,len(a)),random_state=42))
            other=KMeans(n_clusters=k,n_init=10,random_state=123).fit(a)
            rows.append({'space':name,'k':k,'silhouette_train':sil,'stability_ARI':float(adjusted_rand_score(km.labels_,other.labels_)),'inertia':float(km.inertia_)})
            if sil>bestscore:bestscore=sil;best=km
        labels=best.predict(b)
        if name=='embeddings':
            pca=PCA(n_components=2).fit(a);xy=pca.transform(b)
            assignments=pd.DataFrame({'u':np.arange(len(b)),'segment':labels,'pca_1':xy[:,0],'pca_2':xy[:,1]})
    snap=pd.read_parquet(OUT/f'{source}_snapshots.parquet');snap=snap[snap.split=='test'].reset_index(drop=True)
    assignments=pd.concat([assignments,snap],axis=1);assignments['clv_pred']=net.predict(d['x_test'])
    # Acciones sugeridas a partir de perfiles observados, no etiquetas latentes.
    profiles=assignments.groupby('segment').agg(customers=('u','size'),frequency=('frequency','mean'),monetary=('monetary','mean'),recency=('recency','mean'),clv_pred=('clv_pred','mean')).reset_index()
    profiles['name']=profiles.segment.map(lambda k:f'Perfil {k+1}')
    profiles['action']=np.where(profiles.recency>profiles.recency.median(),'Reactivacion por email con consentimiento; KPI: recompra a 90 dias','Venta cruzada personalizada; KPI: conversion a 90 dias')
    assignments.to_parquet(OUT/f'{source}_segments.parquet',index=False)
    profiles.to_csv(REPORT/f'{source}_profiles.csv',index=False);pd.DataFrame(rows).to_csv(REPORT/f'{source}_segmentation.csv',index=False)
    np.save(ART/f'{source}_embeddings.npy',net.encode(d['x_test']))
    return profiles
