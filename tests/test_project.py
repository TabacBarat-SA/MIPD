import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from cliente360.settings import OUT,SYN,RAW,ART
from cliente360.network import MultiTask,sample_negatives
from cliente360.metrics import ranking,regression
from cliente360 import runtime
from cliente360.data import snapshot,load_orders

def test_identity_and_keys():
    c=pd.read_csv(SYN/'olist_customers_dataset.csv');o=pd.read_csv(SYN/'olist_orders_dataset.csv');i=pd.read_csv(SYN/'olist_order_items_dataset.csv');p=pd.read_csv(SYN/'olist_products_dataset.csv')
    assert c.customer_id.is_unique and o.order_id.is_unique
    assert set(o.customer_id)<=set(c.customer_id)
    assert set(i.order_id)<=set(o.order_id)
    assert set(i.product_id)<=set(p.product_id)
    assert not i.duplicated(['order_id','order_item_id']).any()

def test_synthetic_amounts_and_dates():
    ids=set(pd.read_csv(SYN/'provenance.csv').order_id)
    it=pd.read_csv(SYN/'olist_order_items_dataset.csv');it=it[it.order_id.isin(ids)]
    pay=pd.read_csv(SYN/'olist_order_payments_dataset.csv');pay=pay[pay.order_id.isin(ids)]
    amount=(it.price+it.freight_value).groupby(it.order_id).sum()
    paid=pay.groupby('order_id').payment_value.sum().reindex(amount.index)
    assert np.allclose(amount,paid,atol=.011)
    o=pd.read_csv(SYN/'olist_orders_dataset.csv');o=o[o.order_id.isin(ids)]
    for a,b in [('order_purchase_timestamp','order_approved_at'),('order_approved_at','order_delivered_carrier_date'),('order_delivered_carrier_date','order_delivered_customer_date')]:
        assert (pd.to_datetime(o[a])<=pd.to_datetime(o[b])).all()

def test_original_records_preserved():
    for name,key in [('customers','customer_id'),('orders','order_id'),('order_items',None),('order_payments',None)]:
        a=pd.read_csv(RAW/f'olist_{name}_dataset.csv',dtype=str).fillna('')
        b=pd.read_csv(SYN/f'olist_{name}_dataset.csv',dtype=str).fillna('').iloc[:len(a)]
        pd.testing.assert_frame_equal(a,b)

def test_no_future_feature_leakage():
    # Cambiar el futuro no puede alterar las features del corte.
    o=load_orders(SYN);users=runtime.customers()[:20];products=pd.read_parquet(OUT/'products.parquet');cats=sorted(products.category.unique())
    items=pd.read_csv(SYN/'olist_order_items_dataset.csv').merge(o[['order_id','customer_unique_id','order_purchase_timestamp']],on='order_id').merge(products[['product_id','category']],on='product_id',how='left');items.category=items.category.fillna('other')
    x,y,_=snapshot(o,items,users,cats,'2017-10-01')
    changed=o.copy();changed.loc[changed.order_purchase_timestamp>='2017-10-01','amount']*=100
    x2,y2,_=snapshot(changed,items,users,cats,'2017-10-01')
    assert np.array_equal(x,x2)
    assert np.any(y!=y2)

def test_negative_sampling():
    positives=np.array([[0,0],[1,1]]);history=np.array([[0,1],[1,2]])
    pairs,labels=sample_negatives(positives,history,5,3)
    assert not set(map(tuple,pairs[labels==0]))&set(map(tuple,np.vstack([positives,history])))

def test_metrics():
    scores=np.array([[4.,3.,2.,1.]])
    result=ranking(scores,np.array([[0,0],[0,1]]),2)
    assert result['Precision@2']==1 and result['Recall@2']==1 and result['NDCG@2']==1
    assert regression(np.array([1.,2.]),np.array([1.,2.]))['RMSE']==0

def test_saved_model():
    d=np.load(OUT/'synthetic.npz');net=MultiTask.load(ART/'synthetic_multi.npz')
    assert np.isfinite(net.predict(d['x_test'])).all()
    assert (net.predict(d['x_test'])>=0).all()
    assert net.scores(d['x_test']).shape==(len(d['x_test']),len(d['category_idx']))

def test_api_contracts():
    from services.model_api import app
    client=TestClient(app);cid=runtime.customers()[0]
    response=client.get('/customers/'+cid);assert response.status_code==200
    assert len(response.json()['recommendations'])==10
    assert client.get('/customers/unknown').status_code==404
    assert client.get('/customers/'+cid+'?source=invalid').status_code==422
    assert client.get('/customers/'+cid+'?k=10000').status_code==422

def test_other_services():
    from services.data_api import app as data
    from services.segmentation_api import app as seg
    from services.evaluation_api import app as ev
    for app,path in [(data,'/summary'),(data,'/customers'),(seg,'/segments'),(ev,'/evaluation')]:
        c=TestClient(app);assert c.get(path).status_code==200;assert c.get('/health').status_code==200

def test_torch_parity_if_installed():
    torch=pytest.importorskip('torch')
    from cliente360.torch_backend import Cliente360Net
    d=np.load(OUT/'synthetic.npz');ref=MultiTask.load(ART/'synthetic_multi.npz');net=Cliente360Net(ref)
    pairs=torch.tensor([[0,0],[1,2]])
    with torch.no_grad():clv,rec=net(torch.tensor(d['x_test']),pairs)
    np.testing.assert_allclose(clv.numpy(),ref.encode(d['x_test'])@ref.p['w']+ref.p['cb'][0],atol=1e-5)
    np.testing.assert_allclose(rec.numpy(),ref.scores(d['x_test'])[[0,1],[0,2]],atol=1e-5)
