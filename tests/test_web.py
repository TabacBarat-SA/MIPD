"""Contratos de la API de la web 360 Analytics sobre los resultados guardados."""
import pytest
from fastapi.testclient import TestClient
from services.web import app

client = TestClient(app)


@pytest.mark.parametrize('source', ['synthetic', 'original'])
def test_pages_endpoints_return_valid_json(source):
    for path in ['overview', 'groups', 'customers', 'evaluation', 'campaign?limit=5']:
        sep = '&' if '?' in path else '?'
        r = client.get(f'/api/{path}{sep}source={source}')
        assert r.status_code == 200, (path, r.text[:200])


def test_overview_counts_match_profiles():
    d = client.get('/api/overview').json()
    assert sum(p['customers'] for p in d['profiles']) == d['customers'] == len(d['points'])


def test_customer_known_and_unknown():
    first = client.get('/api/customers').json()[0]
    ok = client.get(f"/api/customer/{first['id']}")
    assert ok.status_code == 200 and ok.json()['label'] == first['label']
    assert client.get('/api/customer/desconocido').status_code == 404


def test_campaign_limit_and_csv():
    p = client.get('/api/campaign?limit=3&cost=2').json()
    assert p['selected'] == 3 and p['contact_cost'] == 6
    csv = client.get('/api/campaign.csv?limit=3')
    assert csv.status_code == 200 and csv.text.count('\n') == 4
    assert client.get('/api/campaign?limit=0').status_code == 422


def test_invalid_source_rejected():
    assert client.get('/api/overview?source=otro').status_code == 422
