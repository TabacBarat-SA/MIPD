import json
import numpy as np
import pandas as pd
from .settings import RAW,SYN,OUT,CUTS
FEATURES=['recency','frequency','monetary','ticket','tenure','mean_gap','category_count','installments']

def load_orders(source):
    c=pd.read_csv(source/'olist_customers_dataset.csv')
    o=pd.read_csv(source/'olist_orders_dataset.csv',parse_dates=['order_purchase_timestamp'])
    p=pd.read_csv(source/'olist_order_payments_dataset.csv').groupby('order_id',as_index=False).agg(amount=('payment_value','sum'),installments=('payment_installments','mean'))
    return o[o.order_status=='delivered'].merge(c,on='customer_id',validate='many_to_one').merge(p,on='order_id',validate='one_to_one')

def snapshot(orders,interactions,users,categories,cut):
    cut=pd.Timestamp(cut); end=cut+pd.DateOffset(months=3)
    past=orders[orders.order_purchase_timestamp<cut]
    future=orders[(orders.order_purchase_timestamp>=cut)&(orders.order_purchase_timestamp<end)]
    g=past.groupby('customer_unique_id').agg(frequency=('order_id','nunique'),monetary=('amount','sum'),ticket=('amount','mean'),first=('order_purchase_timestamp','min'),last=('order_purchase_timestamp','max'),installments=('installments','mean'))
    g=g.reindex(users)
    g['recency']=(cut-g['last']).dt.total_seconds()/86400
    g['tenure']=(cut-g['first']).dt.total_seconds()/86400
    g['mean_gap']=(g.tenure-g.recency)/np.maximum(g.frequency-1,1)
    hist=interactions[interactions.order_purchase_timestamp<cut]
    hist=hist[hist.customer_unique_id.isin(users)]
    counts=pd.crosstab(hist.customer_unique_id,hist.category).reindex(index=users,columns=categories,fill_value=0).fillna(0)
    g['category_count']=(counts>0).sum(axis=1)
    catfeatures=counts.div(counts.sum(axis=1).clip(lower=1),axis=0).to_numpy()
    features=np.column_stack([np.log1p(g[FEATURES].fillna(0).to_numpy()),catfeatures]).astype('float32')
    target=future.groupby('customer_unique_id').amount.sum().reindex(users,fill_value=0).to_numpy(dtype='float32')
    frame=g.reset_index().rename(columns={'index':'customer_unique_id'}).drop(columns=['first','last'])
    frame['clv_future']=target;frame['cut']=str(cut.date())
    return features,target,frame

def prepare(source='synthetic'):
    folder=SYN if source=='synthetic' else RAW
    orders=load_orders(folder)
    cohort=pd.read_csv(SYN/'cohort.csv').customer_unique_id.tolist()
    catalog=pd.read_csv(SYN/'catalog.csv').product_id.tolist()
    products=pd.read_csv(folder/'olist_products_dataset.csv').set_index('product_id').reindex(catalog).reset_index()
    products['category']=products.product_category_name.fillna('unknown')
    categories=sorted(products.category.unique()); products['category_idx']=products.category.map({c:i for i,c in enumerate(categories)})
    items=pd.read_csv(folder/'olist_order_items_dataset.csv')
    inter=items.merge(orders[['order_id','customer_unique_id','order_purchase_timestamp']],on='order_id',validate='many_to_one').merge(products[['product_id','category']],on='product_id',how='left',validate='many_to_one')
    inter['category']=inter.category.fillna('other')
    uidx={u:i for i,u in enumerate(cohort)};pidx={p:i for i,p in enumerate(catalog)}
    inter['u']=inter.customer_unique_id.map(uidx);inter['p']=inter.product_id.map(pidx)
    arrays={};frames=[]
    for split,cut in CUTS.items():
        x,y,frame=snapshot(orders,inter,cohort,categories,cut)
        frame['split']=split;frames.append(frame)
        end=pd.Timestamp(cut)+pd.DateOffset(months=3)
        future=inter[(inter.order_purchase_timestamp>=cut)&(inter.order_purchase_timestamp<end)].dropna(subset=['u','p'])
        pairs=future[['u','p']].drop_duplicates().to_numpy(dtype='int64')
        hist=inter[inter.order_purchase_timestamp<cut].dropna(subset=['u','p'])[['u','p']].drop_duplicates().to_numpy(dtype='int64')
        arrays.update({f'x_{split}':x,f'y_{split}':y,f'pairs_{split}':pairs,f'history_{split}':hist})
    # Scaler exclusivamente train. Categorias ya son proporciones.
    mean=arrays['x_train'].mean(axis=0);std=arrays['x_train'].std(axis=0).clip(.05)
    for split in CUTS: arrays[f'x_{split}']=(arrays[f'x_{split}']-mean)/std
    arrays.update(mean=mean,std=std,category_idx=products.category_idx.to_numpy(dtype='int64'))
    attrs=['product_weight_g','product_length_cm','product_height_cm','product_width_cm','product_photos_qty']
    a=np.log1p(products[attrs].fillna(0).to_numpy(dtype='float32'));a=(a-a.mean(0))/a.std(0).clip(.05)
    arrays['product_attributes']=a.astype('float32')
    np.savez_compressed(OUT/f'{source}.npz',**arrays)
    pd.concat(frames).to_parquet(OUT/f'{source}_snapshots.parquet',index=False)
    products.to_parquet(OUT/'products.parquet',index=False)
    (OUT/'mapping.json').write_text(json.dumps({'users':cohort,'products':catalog,'categories':categories,'feature_names':FEATURES+[f'category_{c}' for c in categories]},indent=2))
    return arrays
if __name__=='__main__': prepare()
