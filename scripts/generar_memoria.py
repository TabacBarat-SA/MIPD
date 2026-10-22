"""Genera una memoria de resultados actuales sin narracion fija de una ejecucion anterior.
Ejecutar desde la raiz con uv run python scripts/generar_memoria.py.
"""
from pathlib import Path
import sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cliente360 import business as biz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPORT=ROOT/'reports'
source='synthetic'
meta=json.loads((ROOT/'artifacts'/f'{source}_training.json').read_text())
clv=pd.concat([pd.read_csv(REPORT/f'{source}_{kind}_clv.csv') for kind in ['neural','baseline']]);clv=clv[clv.split=='test'].drop(columns=['source','split'])
rec=pd.concat([pd.read_csv(REPORT/f'{source}_{kind}_rec.csv') for kind in ['neural','baseline']]);rec=rec[rec.split=='test'].drop(columns=['source','split'])
profiles=biz.profiles(source)
audit=json.loads((REPORT/'synthetic_audit.json').read_text())
for name,frame,column,title,label in [
    ('clv_comparison',clv,'RMSE','Error de gasto futuro · evaluación final','Error RMSE (BRL)'),
    ('recommendation_comparison',rec,'NDCG@10','Calidad de recomendación · evaluación final','NDCG@10'),
]:
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(frame.model,frame[column]);ax.set_ylabel(label);ax.set_title(title);ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(REPORT/f'{name}.png',dpi=140);plt.close(fig)
main=clv[clv.model=='multi'].iloc[0];best=clv.sort_values('RMSE').iloc[0];bestrec=rec.sort_values('NDCG@10',ascending=False).iloc[0]
text=f'''# Cliente 360 · Memoria de resultados

## Resumen y procedencia

Herramienta académica para estimar gasto a tres meses, recomendar productos y segmentar clientes a partir de una representación compartida. El escenario de esta memoria es **semisintético**. No se presenta como evidencia real de comportamiento de Olist ni de eficacia publicitaria.

Motor registrado en los pesos: **{meta['backend']}**. Dispositivo registrado: **{meta.get('device','no registrado en esta ejecución inicial')}**. La configuración instalada no cambia la procedencia de los pesos. Esta memoria se construye con los informes guardados; deben corresponder a los mismos pesos y a la misma cohorte.

## Datos y protocolo

Se conservan {audit['original_orders']:,} pedidos originales y se añaden {audit['added_orders']:,} artificiales. La cohorte tiene {audit['cohort_customers']:,} personas y el catálogo reducido {audit['catalog_products']:,} productos. Las variables latentes del generador no son entradas del modelo.

Train: corte 01/10/2017 y etiquetas hasta 01/01/2018. Validación: corte 01/01/2018 y etiquetas hasta 01/04/2018. Test: corte 01/04/2018 y etiquetas hasta 01/07/2018. Las entradas son anteriores a cada corte. Las etiquetas tienen inicio incluido y final excluido. Las mismas personas aparecen en distintos cortes; no es evaluación inductiva de clientes nuevos.

## Arquitectura

Una representación tanh combina embedding de cliente y proyección de variables históricas. La representación de producto combina ID, categoría y atributos físicos. Una cabeza estima gasto y otra calcula afinidades de producto. La pérdida combina MSE y BCE con negative sampling. La segmentación es posterior: K-Means sobre los embeddings, no una tercera cabeza supervisada.

El backend PyTorch permite auto/CPU/CUDA y mueve parámetros y tensores al dispositivo. La validación y el runtime siguen en NumPy. La configuración principal y su ponderación se registran en artifacts; la exploración OAT está en reports. No se afirma que la configuración sea un óptimo global ni que CUDA se haya ejecutado por el mero hecho de estar configurado.

## Evaluación de gasto

{clv.to_markdown(index=False,floatfmt='.4f')}

![Error de gasto](../reports/clv_comparison.png)

En esta tabla, el menor RMSE corresponde a **{best['model']}** ({best.RMSE:.2f} BRL). La red conjunta registra {main.RMSE:.2f} BRL. No se concluye superioridad universal del aprendizaje conjunto. MAE y RMSE miden error monetario; Spearman y lift describen ordenación y concentración de gasto, no efecto causal de publicidad.

## Evaluación de recomendación

{rec.to_markdown(index=False,floatfmt='.4f')}

![Calidad de recomendación](../reports/recommendation_comparison.png)

El mayor NDCG@10 de esta tabla corresponde a **{bestrec['model']}** ({bestrec['NDCG@10']:.4f}). El ranking considera el catálogo reducido completo y permite recompra de productos. Solo entran clientes con compras relevantes futuras; el denominador aparece en la tabla. Los scores no son probabilidades calibradas.

## Segmentos y acciones propuestas

{profiles[['name','customers','frequency','recency','clv_pred','preferred_categories','action','channel','kpi']].to_markdown(index=False,floatfmt='.2f')}

Las categorías preferidas se calculan con artículos comprados antes del corte. Los nombres comerciales se asignan mediante reglas descriptivas de gasto y recencia; no son etiquetas entrenadas de fidelidad o abandono. Las acciones y canales son propuestas para contrastar, no eficacia demostrada. Silueta y estabilidad están en los informes de segmentación; no prueban por sí solas utilidad comercial.

## Aplicación a marketing digital

El panel permite consultar clientes, revisar grupos, proponer personalización y construir una audiencia histórica descargable. El objetivo de campaña se guarda como contexto; el ranking de audiencia se basa en gasto previsto y filtros del usuario, no en una estimación de efecto de tratamiento.

No existen datos de emails, consentimiento, anuncios, clics o campañas ejecutadas. El coste de contacto es un supuesto introducido por el usuario. El gasto previsto no es ingreso atribuible a publicidad. La comparación top CLV frente a azar es retrospectiva, no una estimación causal ni un ROI.

## Discusión y límites

La ampliación incorpora relaciones por diseño; aprenderlas no valida esos supuestos en el mundo real. La muestra original tiene muy pocas compras relevantes en el catálogo reducido, por lo que no permite una conclusión general sobre todo Olist. El modelo rechaza clientes desconocidos en lugar de inventar embeddings.

La variante por categorías agrega salidas de producto; no es una red independiente. La recompra usa clasificadores de referencia; no una cabeza adicional multi-tarea. Los estratos de historial escaso no equivalen a clientes o productos completamente nuevos. La construcción Docker y la ejecución CUDA requieren comprobación en el equipo destino.

## Documentación complementaria

- [Metodología](metodologia.md).
- [Experimentos adicionales v2](anexo_v2.md).
- [Cobertura frente a la propuesta](cobertura.md).
- [Guía de marketing](guia_marketing.md).

La memoria se genera con los resultados actuales; antes de presentar, revisar interpretación, correspondencia entre pesos e informes y evidencia de las pruebas realizadas. No se incluye un PowerPoint terminado ni se garantiza una calificación.
'''
(ROOT/'docs/memoria.md').write_text(text)
parts=['# Anexo v2 · Experimentos adicionales\n\nEscenario semisintético de los informes actuales. No evidencia real de Olist. Motor de los pesos: '+meta['backend']+'.\n']
for suffix,title,note in [
    ('category_rec','Versión por categorías','Media de logits de productos por categoría, sin entrenamiento independiente. Se compara con popularidad y se evalúa con K=3 sobre todas las categorías del catálogo reducido. No se interpreta como probabilidad.'),
    ('sparse_products','Productos con pocas interacciones','Estratos por pares cliente-producto de train. Se restringen los positivos evaluados al estrato, manteniendo todos los productos como candidatos. Cero casos no permite una conclusión.'),
    ('repurchase','Target binario de recompra','Etiqueta gasto futuro>0. Regresión logística y árboles potenciados independientes frente a tasa train. No es una cabeza adicional de la red ni probabilidad de responder a publicidad.'),
]:
    path=REPORT/f'{source}_{suffix}.csv'
    if path.exists():
        f=pd.read_csv(path);f=f[f.split=='test'].drop(columns=['source','split'])
        parts.append('## '+title+'\n\n'+note+'\n\n'+f.to_markdown(index=False,floatfmt='.4f')+'\n')
parts.append('## Lectura responsable\n\nLas comparaciones corresponden al escenario y los informes guardados. No se comparan directamente escalas de gasto, clasificación y ranking como si fueran la misma métrica. No se ha demostrado generalización a clientes o productos desconocidos. La interpretación comercial necesita una prueba adicional con datos válidos de campaña.\n')
(ROOT/'docs/anexo_v2.md').write_text('\n'.join(parts))
print('Memoria, anexo y gráficos actualizados desde los informes guardados.')
