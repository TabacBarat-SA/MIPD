"""Panel de decisiones de marketing. UI sin colores de texto/fondo forzados.
El tema nativo controla widgets, tablas, avisos y contraste claro/oscuro.
"""
from pathlib import Path
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from cliente360 import runtime,business as biz
from cliente360.settings import ART,REPORT,OUT

st.set_page_config(page_title='Cliente 360 | Decisiones de marketing',page_icon='◉',layout='wide')
st.markdown('''<style>
.block-container {max-width:1440px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:16px}
[data-testid="stMetricValue"] {font-size:1.65rem}
[data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.2)}
[data-testid="stCaptionContainer"] {line-height:1.55}
</style>''',unsafe_allow_html=True)
API=os.environ.get('API_URL','').rstrip('/')
@st.cache_data(ttl=120,show_spinner=False)
def query(kind,source,customer_id='',k=10,exclude_seen=False):
    if API:
        params={'source':source,'k':k,'exclude_seen':str(exclude_seen).lower()}
        path=f'/customers/{customer_id}' if kind=='customer' else (f'/history/{customer_id}' if kind=='history' else '/'+kind)
        response=httpx.get(API+path,params=params,timeout=60);response.raise_for_status();return response.json()
    if kind=='customer':return runtime.customer(customer_id,source,k,exclude_seen)
    if kind=='history':return runtime.history(customer_id,source)
    return getattr(runtime,kind)(source)

PALETTE=['#0B6E75','#4F46A5','#C46A14','#2876B8','#AA4768','#527D36','#75604A']
MODEL_NAMES={'multi':'Red conjunta · herramienta','clv_only':'Red especializada en gasto','rec_only':'Red especializada en recomendación','Media':'Referencia: gasto medio','Regresion lineal':'Regresión lineal','Random Forest':'Bosque aleatorio','Gradient Boosting':'Árboles potenciados','Popularidad':'Productos populares','SVD':'Filtrado colaborativo (SVD)'}

def chart(fig):
    # Streamlit aplica el tema activo: no fijar texto blanco sobre fondo claro.
    if any(trace.type=='histogram' for trace in fig.data):fig.update_yaxes(title_text='Número de clientes')
    fig.update_layout(font={'size':14},margin={'l':12,'r':12,'t':48,'b':20},legend_title_text='',height=390)
    st.plotly_chart(fig,width="stretch",theme='streamlit',config={'displaylogo':False})

def table(frame):st.dataframe(frame,hide_index=True,width="stretch")

def download(label,frame,name):
    st.download_button(label,frame.to_csv(index=False).encode('utf-8-sig'),name,'text/csv',width="stretch")

def metric_row(values):
    for col,(label,value,help_text) in zip(st.columns(len(values)),values):
        col.metric(label,value,help=help_text)

def business_names(source):return biz.profiles(source).set_index('segment').name.to_dict()

with st.sidebar:
    st.title('Cliente 360')
    st.caption('DECISIONES DE MARKETING')
    scenario=st.selectbox('Datos que quieres explorar',['Demostración con compras simuladas','Datos originales · misma muestra'])
    source='synthetic' if scenario.startswith('Demostración') else 'original'
    page=st.radio('¿Qué necesitas hacer?',['Resumen del negocio','Conocer a un cliente','Entender los grupos','Preparar una campaña','Comprobar los resultados','Cómo funciona','Guía de uso'])
    st.divider()
    st.caption('Demostración histórica · 01/04/2018\n\nPrevisión: próximos 3 meses.\n\nMoneda: real brasileño (BRL). No se convierten importes a euros.')
    if st.button('Actualizar datos del panel',width="stretch"):
        st.cache_data.clear()
        for fn in [runtime.bundle,runtime.inference,runtime.all_history,biz.audience,biz.profiles,biz.historical_categories]:fn.cache_clear()
        st.rerun()
    st.caption('No envía mensajes ni activa anuncios. Las acciones de campaña son propuestas para revisar.')

st.title(page)
if source=='synthetic':
    st.info('Modo demostración: incluye compras artificiales identificadas. Sirve para aprender y probar decisiones, no para acreditar resultados reales de Olist.')
else:
    st.info('Datos originales sin compras añadidas. Esta muestra tiene poca recompra: las comparaciones con pocos casos no permiten conclusiones fiables.')

try:
    with st.spinner('Cargando los resultados guardados…'):
        summary=query('summary',source)
        people=biz.audience(source)
        profiles=biz.profiles(source)
        names=business_names(source)
    if page=='Resumen del negocio':
        st.write('Decide a quién prestar atención, qué ofrecer y cómo comprobar si la propuesta funciona. Empieza por una de las tres decisiones de abajo.')
        metric_row([
            ('Clientes analizados',biz.number(len(people)),'Personas de la muestra, no todos los clientes de Olist.'),
            ('Gasto previsto · 3 meses',biz.money(people.clv_pred.sum()),'Suma de estimaciones del modelo. No son ventas garantizadas ni ingresos generados por publicidad.'),
            ('Productos disponibles',biz.number(summary['products']),'Catálogo reducido utilizado por el recomendador.'),
            ('Grupos de clientes',str(len(profiles)),'Grupos obtenidos a partir de la representación aprendida por la red.'),
        ])
        st.subheader('Tres preguntas, una herramienta')
        for col,title,body in zip(st.columns(3),['1 · ¿A quién dirigirme?','2 · ¿Qué ofrecerle?','3 · ¿Cómo medirlo?'],[
            'Consulta los grupos y prepara una audiencia ordenada por gasto previsto. No confundas prioridad de contacto con probabilidad de respuesta.',
            'Abre la ficha de un cliente y revisa productos y categorías sugeridos. Contrasta la selección con el historial.',
            'Define un indicador, compara con un grupo de control y revisa la evaluación. Una previsión no demuestra el efecto de una campaña.'
        ]):
            with col:
                with st.container(border=True):st.markdown('**'+title+'**');st.write(body)
        a,b=st.columns(2)
        with a:
            chart(px.histogram(people,x='clv_pred',nbins=30,title='¿Cuánto gasto prevé el modelo por cliente?',labels={'clv_pred':'Gasto previsto a 3 meses (BRL)'},histnorm=None,color_discrete_sequence=PALETTE))
            st.caption('Eje horizontal: importe previsto. Eje vertical: número de clientes. No es una distribución de ventas ya realizadas.')
        with b:
            f=profiles.copy();f['Grupo']=f.segment.map(names)
            chart(px.bar(f,x='customers',y='Grupo',orientation='h',title='¿Cómo se reparte la muestra?',labels={'customers':'Número de clientes'},color_discrete_sequence=PALETTE))
            st.caption('Los nombres describen reglas comerciales sobre cada grupo. No son etiquetas reales de fidelidad ni de abandono.')
        with st.expander('Qué datos estoy viendo y de dónde salen'):
            st.write(f"El experimento conserva {summary['audit']['original_orders']:,} pedidos originales y añade {summary['audit']['added_orders']:,} pedidos artificiales. La muestra del panel tiene {len(people):,} personas.")
            st.write(f"En la ventana histórica de evaluación, el {summary['future_purchase_rate']:.1%} de la muestra realizó alguna compra. Es una tasa observada del escenario, no una probabilidad individual predicha.")
            st.write('Motor registrado en los pesos: '+summary['backend']+'. Instalar PyTorch no cambia el origen de unos pesos ya guardados.')
    elif page=='Conocer a un cliente':
        st.write('Consulta su historial, el gasto previsto y una propuesta de productos. Los alias facilitan la lectura; no son nombres reales.')
        labels=people.set_index('customer_unique_id').client_label.to_dict()
        cid=st.selectbox('Elige un cliente',people.customer_unique_id.tolist(),format_func=lambda x:labels[x])
        with st.expander('Buscar por identificador original'):
            typed=st.text_input('Identificador completo de cliente',help='Debe pertenecer a la muestra. Si no se conoce, no se inventa una predicción.')
            if typed.strip():cid=typed.strip()
        k=st.slider('Cuántos productos quieres revisar',5,20,10)
        exclude=st.checkbox('Mostrar solo productos que no compró antes',False)
        with st.spinner('Consultando la ficha…'):row=query('customer',source,cid,k,exclude)
        st.subheader(labels.get(cid,'Cliente de la muestra'))
        metric_row([
            ('Gasto previsto · 3 meses',biz.money(row['clv_3m_brl']),'Estimación de gasto, no beneficio ni valor de toda la vida.'),
            ('Pedidos anteriores',str(row['frequency']),'Pedidos entregados antes del 01/04/2018.'),
            ('Última compra',f"Hace {row['recency_days']:.0f} días",'Días transcurridos en el corte histórico, no desde hoy.'),
            ('Gasto anterior',biz.money(row['historical_spend_brl']),'Gasto observado antes del corte.'),
        ])
        profile=profiles[profiles.segment==row['segment']].iloc[0]
        with st.container(border=True):
            st.markdown('**Grupo: '+profile['name']+'**')
            st.write('**Acción que podrías probar:** '+profile['action'])
            st.write('**Canal propuesto:** '+profile['channel']+'. **Qué medir:** '+profile['kpi']+'.')
            st.caption('Sugerencia basada en el perfil del grupo, no explicación causal de la predicción individual. Verifica permisos de contacto y condiciones comerciales antes de actuar.')
        recs=row['recommendations']
        st.subheader('Qué productos podrías mostrarle')
        if recs:
            f=pd.DataFrame(recs);f['Orden']=np.arange(1,len(f)+1);f['Categoría']=f.category.map(biz.category_name)
            f['Referencia']=['Producto '+p[:8] for p in f.product_id]
            table(f[['Orden','Referencia','Categoría']])
            st.caption('El orden procede de la afinidad del modelo. Olist no aporta nombres comerciales ni imágenes: no se inventan. El prefijo de referencia es solo una ayuda visual; el ID completo está en la descarga.')
            download('Descargar selección con IDs completos',f[['Orden','product_id','Categoría','affinity_score']],'seleccion_productos.csv')
            with st.expander('Ver afinidades y distribución por categoría'):
                table(f[['Orden','product_id','affinity_score']].rename(columns={'product_id':'ID completo','affinity_score':'Afinidad sin calibrar'}))
                table(biz.category_ranking(recs))
                st.caption('Agrupar la selección por categoría no equivale a haber entrenado y evaluado un recomendador por categorías.')
        else:st.warning('No hay recomendaciones disponibles con señal suficiente en este escenario y selección.')
        st.caption('La evaluación permite recomendar productos ya comprados. La opción de excluirlos cambia esta selección, no las métricas guardadas.')
        st.subheader('Cómo ha comprado antes del corte')
        with st.spinner('Consultando el historial anterior…'):h=pd.DataFrame(query('history',source,cid))
        if len(h):
            h['Fecha']=pd.to_datetime(h.order_purchase_timestamp);h['Origen']=np.where(h.is_synthetic,'Compra simulada','Compra original')
            chart(px.bar(h,x='Fecha',y='amount',color='Origen',title='Historial de pedidos',labels={'amount':'Importe (BRL)'},color_discrete_sequence=PALETTE))
            table(h[['Fecha','amount','Origen']].rename(columns={'amount':'Importe (BRL)'}))
        with st.expander('Identificador y límites de esta ficha'):st.code(cid);st.write(row['warning'])
    elif page=='Entender los grupos':
        st.write('Los clientes se agrupan por similitud. Usa estos perfiles para diseñar mensajes distintos y contrastarlos después con resultados de campaña.')
        for row in profiles.itertuples():
            with st.container(border=True):
                a,b=st.columns([1,2])
                with a:
                    st.subheader(row.name)
                    st.write(f'**{biz.number(row.customers)} clientes** · {row.customers/len(people):.1%} de la muestra')
                    st.write('Gasto previsto medio: **'+biz.money(row.clv_pred)+'**')
                with b:
                    st.write('**Qué observamos:** '+row.reason)
                    st.write('**Categorías más compradas antes del corte:** '+row.preferred_categories)
                    st.write('**Qué probar:** '+row.action)
                    st.write('**Dónde:** '+row.channel+'. **Indicador:** '+row.kpi+'.')
        st.caption('Etiquetas comerciales descriptivas: se asignan con reglas de gasto y días desde la última compra. No se ha entrenado una clasificación de abandono o fidelidad.')
        with st.expander('Ver el mapa de similitud y cómo se eligieron los grupos'):
            points=people.copy();points['Grupo']=points.segment.map(names)
            chart(px.scatter(points,x='pca_1',y='pca_2',color='Grupo',opacity=.6,hover_data=['client_label'],title='Mapa de similitud entre clientes',labels={'pca_1':'Posición en el mapa · eje 1','pca_2':'Posición en el mapa · eje 2','client_label':'Cliente'},color_discrete_sequence=PALETTE))
            st.caption('Cada punto es un cliente. La cercanía resume similitud en una proyección PCA de dos dimensiones; los ejes no son euros, gasto ni riesgo. La proyección pierde información.')
            metrics=pd.read_csv(REPORT/f'{source}_segmentation.csv').rename(columns={'space':'Representación','k':'Número de grupos','silhouette_train':'Separación interna','stability_ARI':'Estabilidad entre semillas'})
            metrics['Representación']=metrics['Representación'].replace({'embeddings':'Representación de la red','RFM':'Historial clásico (RFM)'})
            chart(px.line(metrics,x='Número de grupos',y='Separación interna',color='Representación',markers=True,title='Comparación técnica: red frente a historial clásico',color_discrete_sequence=PALETTE))
            table(metrics)
            st.caption('La silueta mide separación interna, no utilidad comercial demostrada. El número de grupos se elige en entrenamiento; la utilidad de cada acción requiere validación de negocio.')
    elif page=='Preparar una campaña':
        st.write('Construye una audiencia de prueba y descarga una lista revisable. Este módulo no envía emails, no publica anuncios y no dispone de datos de consentimiento.')
        goal=st.selectbox('1 · Objetivo de la prueba',['Venta cruzada personalizada','Reactivación de clientes','Fidelización'])
        st.caption('El objetivo se guarda como contexto. No cambia automáticamente la puntuación: elige los grupos y filtros; la audiencia se ordena por gasto previsto.')
        selected=st.multiselect('2 · Grupos que quieres incluir',profiles.segment.tolist(),default=profiles.segment.tolist(),format_func=lambda x:names[x])
        a,b=st.columns(2)
        with a:min_clv=st.number_input('3 · Gasto previsto mínimo por cliente (BRL)',min_value=0.,value=0.,step=50.)
        with b:limit=st.number_input('4 · Máximo de clientes a seleccionar',min_value=1,max_value=max(1,len(people)),value=min(100,len(people)),step=1)
        use_recency=st.checkbox('Limitar a clientes que compraron en los últimos N días',False)
        max_recency=st.number_input('Máximo de días desde la última compra',min_value=0,value=90,step=10) if use_recency else None
        cost=st.number_input('Coste supuesto por contacto (BRL)',min_value=0.,value=0.,step=.1,help='Introduce un supuesto propio. El sistema no conoce costes reales ni calcula efecto incremental.')
        plan=biz.campaign_plan(source,selected,min_clv,max_recency,int(limit),cost)
        metric_row([
            ('Cumplen los filtros',biz.number(plan['eligible']),'Clientes disponibles antes de aplicar el máximo.'),
            ('Audiencia seleccionada',biz.number(plan['selected']),'Ordenada por gasto previsto de mayor a menor.'),
            ('Gasto previsto de la audiencia',biz.money(plan['predicted_spend']),'No es facturación atribuible a esta campaña.'),
            ('Coste supuesto de contacto',biz.money(plan['contact_cost']),'Número seleccionado multiplicado por el coste introducido.'),
        ])
        st.warning('El gasto previsto no es el ingreso que generará la campaña. No se calcula ROI ni aumento de ventas porque no se conoce el efecto causal de contactar a estos clientes.')
        if not plan['selected']:st.info('No hay clientes que cumplan los filtros. Amplía los grupos o reduce el mínimo de gasto previsto.')
        else:
            st.subheader('Audiencia priorizada para revisar')
            preview=plan['export'][['alias_interfaz','grupo','gasto_estimado_3_meses_brl','dias_desde_ultima_compra']].rename(columns={'alias_interfaz':'Cliente','grupo':'Grupo','gasto_estimado_3_meses_brl':'Gasto previsto (BRL)','dias_desde_ultima_compra':'Días desde última compra'})
            table(preview.head(100).round(2))
            if len(preview)>100:st.caption('Se muestran los primeros 100. La descarga incluye toda la audiencia seleccionada.')
            st.caption('Los IDs son históricos y anonimizados. No hay emails ni permisos de contacto: esta lista no es una audiencia lista para subir a una plataforma publicitaria.')
            confirm=st.checkbox('Entiendo que es una simulación histórica y que debo revisar los permisos y la selección antes de cualquier uso real.')
            if confirm:
                export=plan['export'].copy();export['objetivo_propuesto']=goal
                download('Descargar audiencia de demostración',export,'audiencia_demostracion.csv')
        with st.container(border=True):
            st.subheader('Cómo comprobar la idea de marketing')
            st.write('**Hipótesis propuesta:** una oferta adaptada al perfil puede mejorar la respuesta frente a una comunicación genérica. No es un resultado demostrado.')
            st.write('**Diseño propuesto:** dentro de una audiencia válida, asignar al azar un grupo al mensaje personalizado y otro a una referencia o control. Fijar antes el indicador, el periodo y las condiciones de la oferta.')
            st.write('**Qué registrar:** personas asignadas, mensajes entregados, clics, compras, gasto, costes y bajas. Comparar tasas sobre el mismo denominador y revisar la incertidumbre antes de concluir.')
            st.caption('El proyecto no ejecuta esta prueba. La asignación, los resultados y el consentimiento no están en los datos disponibles.')
    elif page=='Comprobar los resultados':
        st.write('Comprueba si la herramienta aporta valor frente a alternativas más sencillas. Una red más compleja no tiene por qué ser mejor.')
        e=query('evaluation',source)
        split_label=st.selectbox('Qué periodo quieres revisar',['Evaluación final (test)','Ajuste del modelo (validación)'])
        split='test' if split_label.startswith('Evaluación') else 'val'
        c=pd.DataFrame(e['clv']);c=c[c.split==split].copy();c['Modelo']=c.model.map(MODEL_NAMES).fillna(c.model)
        r=pd.DataFrame(e['recommendation']);r=r[r.split==split].copy();r['Modelo']=r.model.map(MODEL_NAMES).fillna(r.model)
        t1,t2,t3=st.tabs(['1 · Prever gasto','2 · Recomendar productos','3 · Aprendizaje conjunto y experimentos'])
        with t1:
            st.subheader('¿Qué modelo se equivoca menos al prever el gasto?')
            best=c.sort_values('RMSE').iloc[0];joint=c[c.model=='multi'].iloc[0]
            st.write(f"En este periodo, **{best['Modelo']}** tiene el menor error RMSE: **{biz.money(best.RMSE)}**. La red conjunta registra **{biz.money(joint.RMSE)}**. Es una comparación del escenario, no una garantía comercial.")
            chart(px.bar(c.sort_values('RMSE'),x='RMSE',y='Modelo',orientation='h',title='Error de previsión: una barra menor es mejor',labels={'RMSE':'Error RMSE (BRL)'},color_discrete_sequence=PALETTE))
            table(c[['Modelo','MAE','RMSE','Spearman','lift_top10']].rename(columns={'MAE':'Error absoluto medio (BRL)','RMSE':'Error RMSE (BRL)','Spearman':'Ordenación (Spearman)','lift_top10':'Concentración de gasto · top 10 %'}).round(3))
            st.caption('MAE: error absoluto medio. RMSE penaliza más los errores grandes. Spearman evalúa el orden. El lift compara gasto observado del decil predicho superior con la media; no es incremento causado por publicidad.')
            download('Descargar comparativa de gasto',c,'comparativa_gasto.csv')
        with t2:
            n=int(r.eligible_users.max())
            st.write(f'Clientes con compras futuras relevantes para esta evaluación: **{biz.number(n)}**.')
            if n<30:st.warning('Hay muy pocos clientes evaluables. No interpretes estos resultados como evidencia fiable sobre la calidad del recomendador.')
            else:
                best=r.sort_values('NDCG@10',ascending=False).iloc[0]
                st.write(f"**{best['Modelo']}** obtiene la mejor ordenación NDCG@10 de esta tabla. La herramienta sigue mostrando la red conjunta para estudiar el enfoque compartido, no porque gane necesariamente.")
            chart(px.bar(r,x='NDCG@10',y='Modelo',orientation='h',title='Calidad del orden: una barra mayor es mejor',color_discrete_sequence=PALETTE))
            table(r[['Modelo','Precision@10','Recall@10','NDCG@10','coverage','eligible_users']].rename(columns={'Precision@10':'Aciertos entre 10 sugerencias','Recall@10':'Compras relevantes recuperadas','NDCG@10':'Calidad del orden','coverage':'Cobertura del catálogo','eligible_users':'Clientes evaluables'}).round(4))
            st.caption('Las métricas están en escala 0–1. Solo se incluyen clientes con compras relevantes futuras; el catálogo completo reducido se usa como conjunto de candidatos. No son tasas de conversión de una campaña.')
        with t3:
            st.subheader('¿Ayuda compartir lo aprendido?')
            st.write('Compara «Red conjunta» con las redes especializadas de las otras pestañas. Si una especializada mejora, compartir representación no ha aportado ventaja en esa tarea. Ese resultado es válido para la asignatura.')
            st.subheader('Sensibilidad de configuración · solo validación')
            sensitivity=pd.DataFrame(e['sensitivity'])
            if len(sensitivity):table(sensitivity.rename(columns={'config':'Configuración','epochs':'Épocas ejecutadas'}).round(4))
            else:st.info('No hay una exploración de sensibilidad guardada para este escenario.')
            st.subheader('Historial escaso')
            table(pd.DataFrame(e['sparse_history']).rename(columns={'group':'Historial previo','n':'Clientes'}).round(3))
            st.caption('No equivale a evaluar clientes completamente nuevos.')
            for suffix,title in [('category_rec','Recomendación por categoría · K=3'),('sparse_products','Productos con pocas interacciones'),('repurchase','Target alternativo: recompra en tres meses')]:
                path=REPORT/f'{source}_{suffix}.csv'
                if path.exists():
                    st.subheader(title)
                    extra=pd.read_csv(path);table(extra[extra.split==split].drop(columns=['source','split']).round(4))
            st.caption('La versión por categoría promedia afinidades de productos, no entrena otra red. La recompra se compara con clasificadores independientes, no con una nueva cabeza multi-tarea. No se estima respuesta a publicidad.')
            st.subheader('Selección retrospectiva frente a azar')
            campaign=e['campaign'];metric_row([
                ('Clientes seleccionados',str(campaign['n_targeted']),'Decil superior según el modelo.'),
                ('Gasto observado de la selección',biz.money(campaign['selected_observed_revenue']),'Compras que ocurrieron en la ventana de evaluación.'),
                ('Referencia aleatoria esperada',biz.money(campaign['random_expected_revenue']),'Promedio de selecciones aleatorias del mismo tamaño.'),
            ])
            st.warning('Es una comparación retrospectiva de selección. No demuestra que una campaña haya generado ese gasto ni estima ingresos incrementales.')
            with st.expander('Ver las curvas de aprendizaje del modelo actual'):
                curve=pd.read_csv(REPORT/f'{source}_multi_learning.csv')
                chart(px.line(curve,x='epoch',y=['clv_loss','rec_loss','val_loss'],title='Evolución de las pérdidas',labels={'epoch':'Época','value':'Pérdida','variable':'Serie'},color_discrete_sequence=PALETTE))
                st.caption('Las pérdidas no son porcentajes de ventas ni probabilidades. Su escala depende de la tarea y de su ponderación.')
    elif page=='Cómo funciona':
        st.write('Un mismo historial alimenta tres decisiones. La evaluación técnica permite comprobar si la complejidad está justificada.')
        for title,body in [
            ('1 · Reunir el historial','Se unen clientes, pedidos, artículos, pagos y productos. La identidad persistente es customer_unique_id; customer_id cambia entre pedidos.'),
            ('2 · Aprender una representación compartida','La red resume patrones del cliente y del producto en vectores numéricos. Combina identidad, historial, categoría y atributos físicos.'),
            ('3 · Estimar y recomendar','Una salida estima gasto futuro; otra ordena productos por afinidad. Ambas comparten parte de lo aprendido. La segmentación usa después esa representación, no una tercera cabeza supervisada.'),
            ('4 · Traducir a acciones y comprobar','Los grupos reciben etiquetas comerciales descriptivas. Se proponen acciones, se prepara una audiencia y se compara el modelo con alternativas. La eficacia publicitaria necesita un experimento adicional.'),
        ]:
            with st.container(border=True):st.markdown('**'+title+'**');st.write(body)
        st.subheader('Qué sabe el sistema y qué no sabe')
        table(pd.DataFrame([
            ['Disponible','Historial de pedidos, importes, productos, categorías y atributos.'],
            ['No disponible','Emails, permisos de contacto, exposición a anuncios, clics o resultados de campañas.'],
            ['Estimado','Gasto futuro y orden de afinidad de productos.'],
            ['No demostrado','Probabilidad calibrada de compra, abandono individual, beneficio, ROI o efecto causal de publicidad.'],
        ],columns=['Tipo','Información']))
        with st.expander('Protocolo académico y correspondencia con la propuesta'):
            st.markdown('''- **Train:** corte 01/10/2017; etiquetas hasta 01/01/2018.
- **Validación:** corte 01/01/2018; etiquetas hasta 01/04/2018.
- **Test:** corte 01/04/2018; etiquetas hasta 01/07/2018.
- Las entradas usan compras anteriores a cada corte. Las ventanas de etiquetas no se solapan.
- Escalador, centroides y PCA se ajustan con entrenamiento. Los pesos no se entrenan con etiquetas de test.
- La red combina MSE de gasto y BCE de recomendación, con negativos muestreados y ponderación lambda.
- Se comparan modelos clásicos, tareas aisladas, sensibilidad y segmentación RFM frente a representación de la red.
- Se conserva el resultado aunque la red no mejore a los modelos clásicos.
''')
            st.write('Motor registrado para estos pesos: **'+summary['backend']+'**. La configuración del paquete permite entrenar con PyTorch y CUDA; los pesos de demostración conservan su procedencia real.')
            st.warning('Cobertura parcial de la propuesta: la variante por categoría agrega la salida de producto y la recompra usa clasificadores independientes; no se ha entrenado una cabeza adicional. La validación completa de Docker/CUDA en el equipo destino sigue pendiente. Consulta docs/cobertura.md.')
    else:
        st.subheader('Recorrido recomendado para una persona de marketing')
        st.markdown('''1. **Resumen del negocio:** identifica el tamaño de la muestra y entiende qué se está estimando.
2. **Entender los grupos:** revisa qué diferencia a cada perfil y qué acción podrías probar.
3. **Conocer a un cliente:** contrasta la propuesta de productos con su historial.
4. **Preparar una campaña:** elige objetivo, filtros y tamaño; revisa la audiencia antes de descargarla.
5. **Comprobar los resultados:** verifica si la red aporta algo frente a métodos más sencillos.
''')
        st.subheader('Diccionario sin tecnicismos')
        table(pd.DataFrame([
            ['Gasto previsto / CLV a 3 meses','Estimación de cuánto gastará el cliente en ese periodo; no beneficio ni valor de toda su vida.'],
            ['Días desde la última compra','Tiempo transcurrido en la fecha histórica de corte. En los datos se llama recencia.'],
            ['Grupo / segmento','Conjunto de clientes con patrones parecidos. No demuestra intención de compra o abandono.'],
            ['Afinidad','Puntuación para ordenar productos. No significa un porcentaje de probabilidad.'],
            ['Representación / embedding','Resumen numérico que aprende la red a partir de los datos.'],
            ['Ablación','Comparación de la red conjunta con las tareas entrenadas por separado.'],
            ['Datos semisintéticos','Originales más compras artificiales identificadas. No equivalen a comportamiento real validado.'],
        ],columns=['Concepto','Qué significa']))
        st.subheader('Antes de presentar o utilizar la herramienta')
        st.write('Aclara el escenario, la fecha histórica y la moneda. No atribuyas ventas a una campaña no ejecutada. No presentes afinidades como probabilidades. Muestra también los modelos que superan a la red y los requisitos pendientes.')
        st.caption('Para instalar y entrenar: README.md. Para justificar decisiones: docs/metodologia.md. Para revisar alcance: docs/cobertura.md.')
except (KeyError,ValueError,httpx.HTTPError,FileNotFoundError) as error:
    st.error('No se pudo completar la consulta: '+str(error))
    st.info('Comprueba que los datos, pesos y métricas pertenecen a la misma ejecución. Si acabas de entrenar, ejecuta evaluate y segment, reinicia los servicios y pulsa «Actualizar datos del panel».')
