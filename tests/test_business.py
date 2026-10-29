import numpy as np
import pandas as pd
from cliente360 import business as biz,runtime
from cliente360.settings import REPORT

def test_aliases_and_group_coverage():
    f=biz.audience();p=biz.profiles()
    assert f.client_label.is_unique
    assert int(p.customers.sum())==len(f)
    assert set(p.segment)==set(f.segment)
    assert p.name.is_unique

def test_campaign_filters_and_cost():
    f=biz.audience();group=int(f.segment.iloc[0]);minimum=float(f.clv_pred.quantile(.25))
    result=biz.campaign_plan(segments=[group],min_clv=minimum,max_recency=150,limit=20,cost=.5)
    e=result['export']
    assert result['selected']<=20
    assert (e.gasto_estimado_3_meses_brl>=minimum).all()
    assert (e.dias_desde_ultima_compra<=150).all()
    assert e.gasto_estimado_3_meses_brl.is_monotonic_decreasing
    assert result['contact_cost']==len(e)*.5
    assert np.isclose(result['predicted_spend'],e.gasto_estimado_3_meses_brl.sum())
    assert e.es_simulacion_historica.all()

def test_empty_audience():
    result=biz.campaign_plan(segments=[],limit=100)
    assert result['selected']==0 and result['contact_cost']==0
    assert 'id_cliente' in result['export']

def test_cached_predictions_equal_reference():
    d,net,*_=runtime.bundle();pred,scores=runtime.inference()
    np.testing.assert_allclose(pred,net.predict(d['x_test']))
    np.testing.assert_allclose(scores,net.scores(d['x_test']))

def test_extended_outputs():
    for source in ['synthetic','original']:
        for suffix in ['category_rec','sparse_products','repurchase']:
            f=pd.read_csv(REPORT/f'{source}_{suffix}.csv')
            assert set(f.split)=={'val','test'}
        f=pd.read_csv(REPORT/f'{source}_category_rec.csv')
        assert f['NDCG@3'].between(0,1).all()
        f=pd.read_csv(REPORT/f'{source}_repurchase.csv')
        assert f.Brier.between(0,1).all()

def test_category_translation():
    assert biz.category_name('esporte_lazer')=='Deporte y ocio'

def test_money_falls_back_to_reais_without_exchange_rate(monkeypatch):
    import forex_python.converter as fx
    def offline(*a,**k):raise ConnectionError('sin red')
    monkeypatch.setattr(fx.CurrencyRates,'convert',offline)
    assert biz.money(1234.5)=='R$ 1.234,50'

def test_historical_category_shares():
    f=biz.historical_categories()
    assert f.share.between(0,1).all()
    assert (f.groupby('segment').size()<=3).all()
    assert f.groupby('segment').share.sum().le(1.000001).all()

def test_torch_trainer_cpu_if_installed(monkeypatch):
    import pytest
    pytest.importorskip('torch')
    from cliente360.network import MultiTask
    from cliente360.torch_backend import TorchTrainer
    monkeypatch.setenv('CLIENTE360_DEVICE','cpu')
    ref=MultiTask(3,4,2,np.array([0,0,1,1]),np.zeros((4,2),dtype='float32'))
    trainer=TorchTrainer(ref)
    x=np.zeros((3,2),dtype='float32');y=np.ones(3,dtype='float32');pairs=np.array([[0,0],[1,1]],dtype='int64');labels=np.array([1.,0.],dtype='float32')
    losses=trainer.update(x,y,pairs,labels,1.,1.);trainer.export(ref)
    assert trainer.device.type=='cpu' and np.isfinite(losses).all()
    assert np.isfinite(ref.scores(x)).all()

def test_torch_cuda_if_available(monkeypatch):
    import pytest
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('GPU CUDA no disponible')
    from cliente360.network import MultiTask
    from cliente360.torch_backend import TorchTrainer
    monkeypatch.setenv('CLIENTE360_DEVICE','cuda')
    ref=MultiTask(3,4,2,np.array([0,0,1,1]),np.zeros((4,2),dtype='float32'))
    trainer=TorchTrainer(ref)
    losses=trainer.update(np.zeros((3,2),dtype='float32'),np.ones(3,dtype='float32'),np.array([[0,0],[1,1]]),np.array([1.,0.],dtype='float32'),1.,1.)
    assert trainer.device.type=='cuda' and np.isfinite(losses).all()
