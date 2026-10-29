"""API de la interfaz web 360 Analytics y servidor del front compilado.
Lee los resultados guardados mediante runtime y business. No reentrena nada.
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from cliente360 import business as biz, runtime
from cliente360.settings import REPORT

SOURCE = Query('synthetic', pattern='^(synthetic|original)$')
DIST = Path(__file__).resolve().parents[1] / 'web' / 'dist'
app = FastAPI(title='360 Analytics', docs_url='/api/docs', openapi_url='/api/openapi.json')


def clean(value):
    """Convierte tipos de numpy/pandas y NaN/inf a JSON valido."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return clean(value.to_dict('records'))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (pd.Timestamp, np.bool_)):
        return str(value) if isinstance(value, pd.Timestamp) else bool(value)
    return value


def _frame(frame):
    return json.loads(frame.to_json(orient='records'))


@app.get('/api/health')
def health():
    return {'status': 'ok', 'service': 'web'}


_fx = {'at': 0.0, 'rate': None}


@app.get('/api/fx')
def fx():
    """Tipo BRL->USD. Si no hay red devuelve null y la web muestra BRL."""
    ttl = 3600 if _fx['rate'] else 300
    if time.time() - _fx['at'] > ttl:
        _fx['at'] = time.time()
        try:
            from forex_python.converter import CurrencyRates
            _fx['rate'] = float(CurrencyRates().get_rate('BRL', 'USD'))
        except Exception:
            _fx['rate'] = _fx['rate'] if _fx['rate'] else None
    return {'BRL_USD': _fx['rate']}


@app.get('/api/overview')
def overview(source: str = SOURCE):
    summary = runtime.summary(source)
    people = biz.audience(source)
    profiles = biz.profiles(source)
    counts, edges = np.histogram(people.clv_pred, bins=28)
    return clean({
        'summary': summary,
        'customers': len(people),
        'predicted_spend': float(people.clv_pred.sum()),
        'products': summary['products'],
        'groups': len(profiles),
        'histogram': [{'from': float(edges[i]), 'to': float(edges[i + 1]), 'count': int(c)} for i, c in enumerate(counts)],
        'profiles': _frame(profiles[['segment', 'name', 'customers', 'clv_pred']]),
        'points': _frame(people[['segment', 'pca_1', 'pca_2', 'clv_pred']]),
    })


@app.get('/api/groups')
def groups(source: str = SOURCE):
    people = biz.audience(source)
    seg = pd.read_csv(REPORT / f'{source}_segmentation.csv')
    return clean({
        'total': len(people),
        'profiles': _frame(biz.profiles(source)),
        'points': _frame(people[['client_label', 'segment', 'pca_1', 'pca_2', 'clv_pred']]),
        'metrics': _frame(seg),
    })


@app.get('/api/customers')
def customers(source: str = SOURCE):
    people = biz.audience(source)
    return clean(_frame(people[['customer_unique_id', 'client_label', 'segment', 'clv_pred']].rename(columns={'customer_unique_id': 'id', 'client_label': 'label'})))


@app.get('/api/customer/{customer_id}')
def customer(customer_id: str, source: str = SOURCE, k: int = Query(10, ge=1, le=20), exclude_seen: bool = False):
    try:
        row = runtime.customer(customer_id, source, k, exclude_seen)
    except KeyError as e:
        raise HTTPException(404, str(e).strip('"\'')) from e
    people = biz.audience(source).set_index('customer_unique_id')
    profile = biz.profiles(source).set_index('segment').loc[row['segment']]
    history = runtime.history(customer_id, source)
    recs = [{**r, 'category_es': biz.category_name(r['category'])} for r in row['recommendations']]
    return clean({
        **row,
        'recommendations': recs,
        'label': people.client_label.get(customer_id),
        'profile': {k_: profile[k_] for k_ in ['name', 'action', 'channel', 'kpi']},
        'history': history,
        'categories': _frame(biz.category_ranking(row['recommendations'])) if recs else [],
    })


def _plan(source, segments, min_clv, max_recency, limit, cost):
    ids = [int(s) for s in segments.split(',') if s.strip() != ''] if segments is not None else None
    try:
        return biz.campaign_plan(source, ids, min_clv, max_recency, limit, cost)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get('/api/campaign')
def campaign(source: str = SOURCE, segments: str | None = None, min_clv: float = Query(0, ge=0), max_recency: float | None = Query(None, ge=0),
             limit: int = Query(100, ge=1), cost: float = Query(0, ge=0)):
    plan = _plan(source, segments, min_clv, max_recency, limit, cost)
    export = plan.pop('export')
    return clean({**plan, 'rows': _frame(export.head(100))})


@app.get('/api/campaign.csv')
def campaign_csv(source: str = SOURCE, segments: str | None = None, min_clv: float = Query(0, ge=0), max_recency: float | None = Query(None, ge=0),
                 limit: int = Query(100, ge=1), cost: float = Query(0, ge=0), goal: str = 'Venta cruzada personalizada'):
    export = _plan(source, segments, min_clv, max_recency, limit, cost)['export']
    export['objetivo_propuesto'] = goal
    buf = io.StringIO()
    export.to_csv(buf, index=False)
    return Response(buf.getvalue().encode('utf-8-sig'), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="audiencia_demostracion.csv"'})


@app.get('/api/evaluation')
def evaluation(source: str = SOURCE):
    e = runtime.evaluation(source)
    extras = {}
    for key, suffix in [('category', 'category_rec'), ('sparse_products', 'sparse_products'), ('repurchase', 'repurchase')]:
        path = REPORT / f'{source}_{suffix}.csv'
        extras[key] = _frame(pd.read_csv(path)) if path.exists() else []
    learning = pd.read_csv(REPORT / f'{source}_multi_learning.csv')
    return clean({**e, **{'extras': extras}, 'learning': _frame(learning)})


@app.get('/api/{rest:path}')
def not_found(rest: str):
    raise HTTPException(404, 'Ruta no encontrada')


if DIST.exists():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')

    @app.get('/{path:path}')
    def spa(path: str):
        target = (DIST / path).resolve()
        if path and target.is_file() and DIST in target.parents:
            return FileResponse(target)
        return FileResponse(DIST / 'index.html')
