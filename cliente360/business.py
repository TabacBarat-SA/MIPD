"""Vistas de negocio sobre resultados existentes. No altera pesos ni metricas.
Las etiquetas comerciales son reglas descriptivas, no clases entrenadas.
"""
from functools import lru_cache
import numpy as np
import pandas as pd
from . import runtime
from .settings import OUT

CATEGORY_ES = {
 'cama_mesa_banho':'Hogar y textiles', 'beleza_saude':'Belleza y cuidado personal',
 'esporte_lazer':'Deporte y ocio', 'moveis_decoracao':'Muebles y decoración',
 'informatica_acessorios':'Informática y accesorios', 'utilidades_domesticas':'Artículos para el hogar',
 'relogios_presentes':'Relojes y regalos', 'telefonia':'Telefonía', 'ferramentas_jardim':'Herramientas y jardín',
 'automotivo':'Automoción', 'brinquedos':'Juguetes', 'perfumaria':'Perfumería',
 'cool_stuff':'Artículos originales', 'bebes':'Bebés', 'papelaria':'Papelería',
 'unknown':'Sin categoría', 'other':'Otras categorías',
}
def category_name(value):
    return CATEGORY_ES.get(str(value),str(value).replace('_',' ').capitalize()+' (categoría original)')

def money(value):
    return ('R$'+f'{float(value):,.2f}').replace(',','X').replace('.',',').replace('X','.')

def number(value):return f'{int(value):,}'.replace(',','.')

@lru_cache(maxsize=2)
def audience(source='synthetic'):
    _,_,mapping,seg,_=runtime.bundle(source)
    result=seg.copy()
    result['client_label']=[f'Cliente {i+1:04d}' for i in range(len(result))]
    # Los alias son de interfaz, no nombres reales de personas.
    if result.customer_unique_id.tolist()!=mapping['users']:
        raise ValueError('Los segmentos y el indice de clientes no corresponden a la misma cohorte.')
    return result

@lru_cache(maxsize=2)
def profiles(source='synthetic'):
    frame=audience(source)
    p=frame.groupby('segment').agg(customers=('u','size'),frequency=('frequency','mean'),monetary=('monetary','mean'),recency=('recency','mean'),clv_pred=('clv_pred','mean')).reset_index()
    q=frame.clv_pred.quantile(.75);r=frame.recency.median()
    names=[];actions=[];channels=[];kpis=[];reasons=[]
    for row in p.itertuples():
        if row.clv_pred>=q and row.recency<=r:
            name='Fidelización prioritaria';action='Probar una selección de productos complementarios, sin descuento generalizado.';channel='Email personalizado o área privada';kpi='Recompra a 90 días y margen por pedido'
        elif row.recency>r:
            name='Reactivación';action='Probar un recordatorio de novedades y preferencias antes de ofrecer un incentivo.';channel='Email con permiso de contacto';kpi='Recompra a 90 días y bajas de suscripción'
        else:
            name='Desarrollo de relación';action='Probar venta cruzada con categorías afines al historial.';channel='Recomendaciones en web o email';kpi='Conversión y valor medio del pedido'
        names.append(f'{name} · G{int(row.segment)+1}')
        actions.append(action);channels.append(channel);kpis.append(kpi)
        reasons.append(f"{row.frequency:.1f} pedidos previos de media y {row.recency:.0f} días desde la última compra; gasto estimado medio {money(row.clv_pred)}.")
    p['name']=names;p['action']=actions;p['channel']=channels;p['kpi']=kpis;p['reason']=reasons
    categories=historical_categories(source)
    text={int(g):'; '.join(f'{row.category_es} ({row.share:.0%})' for row in frame.itertuples()) for g,frame in categories.groupby('segment')}
    p['preferred_categories']=p.segment.map(text).fillna('Sin historial de categorías disponible')
    return p

def select_audience(source='synthetic',segments=None,min_clv=0.,max_recency=None):
    f=audience(source).copy()
    if segments is not None:f=f[f.segment.isin(segments)]
    f=f[f.clv_pred>=float(min_clv)]
    if max_recency is not None:f=f[f.recency<=float(max_recency)]
    return f.sort_values(['clv_pred','customer_unique_id'],ascending=[False,True],kind='stable')

def campaign_plan(source='synthetic',segments=None,min_clv=0.,max_recency=None,limit=100,cost=0.):
    if int(limit)<1 or float(cost)<0:raise ValueError('Límite y coste no válidos.')
    pool=select_audience(source,segments,min_clv,max_recency)
    selected=pool.head(int(limit)).copy()
    names=profiles(source).set_index('segment').name.to_dict()
    export=selected[['customer_unique_id','client_label','segment','clv_pred','recency','frequency']].copy()
    export['segment']=export.segment.map(names)
    export=export.rename(columns={'customer_unique_id':'id_cliente','client_label':'alias_interfaz','segment':'grupo','clv_pred':'gasto_estimado_3_meses_brl','recency':'dias_desde_ultima_compra','frequency':'pedidos_previos'})
    export['escenario']=source;export['fecha_corte']='2018-04-01';export['es_simulacion_historica']=True
    return {'eligible':len(pool),'selected':len(selected),'predicted_spend':float(selected.clv_pred.sum()),'contact_cost':float(len(selected)*cost),'export':export}

def category_ranking(recommendations):
    """Agrupa el ranking de producto; NO es otro recomendador entrenado."""
    if not recommendations:return pd.DataFrame(columns=['Categoría','Productos en la selección'])
    f=pd.DataFrame(recommendations);f['Categoría']=f.category.map(category_name)
    return f.groupby('Categoría',sort=False).size().reset_index(name='Productos en la selección')

@lru_cache(maxsize=2)
def historical_categories(source='synthetic'):
    """Preferencias descriptivas por grupo: solo articulos anteriores al corte."""
    from .settings import SYN,RAW
    folder=SYN if source=='synthetic' else RAW
    history=runtime.all_history(source)[['order_id','customer_unique_id']]
    people=audience(source)[['customer_unique_id','segment']]
    history=history.merge(people,on='customer_unique_id',validate='many_to_one')
    items=pd.read_csv(folder/'olist_order_items_dataset.csv',usecols=['order_id','product_id'])
    products=pd.read_csv(folder/'olist_products_dataset.csv',usecols=['product_id','product_category_name'])
    joined=items.merge(history,on='order_id',validate='many_to_one').merge(products,on='product_id',validate='many_to_one')
    joined['category']=joined.product_category_name.fillna('unknown')
    counts=joined.groupby(['segment','category']).size().reset_index(name='items')
    counts['share']=counts['items']/counts.groupby('segment')['items'].transform('sum')
    counts['category_es']=counts.category.map(category_name)
    return counts.sort_values(['segment','items','category'],ascending=[True,False,True]).groupby('segment').head(3).reset_index(drop=True)
