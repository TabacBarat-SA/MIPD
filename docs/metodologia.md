# Metodología y decisiones

## 1. Por qué no basta con duplicar filas

Copiar compras sin modificar identidad, fechas y relaciones crea duplicados, no recurrencia válida. Aquí se generan pedidos nuevos con claves nuevas, conservando la identidad única de la persona. Los importes de pagos coinciden con la suma de artículos y transporte. Las fechas siguen compra, aprobación, expedición y entrega. Se añaden únicamente pedidos entregados: no se pretende modelar cancelaciones o retrasos reales.

## 2. Generador

Se seleccionan 2.500 personas con una compra entregada anterior al 01/04/2017. El catálogo contiene hasta 30 productos de cada una de las diez categorías más frecuentes antes de esa fecha. Es una reducción deliberada de dispersión, no una representación del catálogo completo.

Para cada persona se toma el ticket y la categoría de su primera compra conocida. La intensidad mensual se obtiene de una lognormal con media logarítmica `-0.45 + 0.20*log(ticket/100)`, dispersión 0,85 y límites 0,08–3,5. El presupuesto es el ticket multiplicado por ruido lognormal y acotado entre 25 y 700 BRL. Un 28 % de los perfiles puede abandonar en un mes aleatorio. Son parámetros de diseño, no estimaciones de la realidad de Olist.

El número de pedidos mensuales es Poisson. Noviembre y diciembre reciben factores de intensidad 1,30 y 1,12. La categoría preferida se elige con probabilidad 0,78; en el resto de ocasiones se explora. Dentro de una categoría, la elección combina proximidad al presupuesto y popularidad histórica anterior a abril de 2017. Precio y transporte incorporan ruido lognormal; las cantidades son 1–3 unidades. El número de cuotas aumenta suavemente con el importe si se paga con tarjeta.

La señal no es determinista: hay compras inesperadas, abandono, clientes sin recompra y variación de importes. No se consulta el target de test para generar registros ni para escoger hiperparámetros. La semilla es 42.

## 3. Ventanas y fuga

Las variables de cada corte proceden de fechas estrictamente anteriores. Las etiquetas son gasto y productos comprados desde el corte hasta tres meses después, con extremo derecho excluido. Las ventanas de etiquetas no se solapan. Las mismas personas aparecen en distintos cortes, lo que mide predicción temporal de clientes conocidos, no generalización a personas desconocidas.

El escalador de clientes se ajusta solo en train. El catálogo se conoce antes de train. Las categorías y atributos físicos son información estática de ese catálogo. El test anti-fuga modifica importes futuros y verifica que las features anteriores no cambian.

## 4. Arquitectura compartida

`z_u = tanh(U_u + X_u W + b)`.

`v_p = tanh(V_p + C_categoria(p) + atributos_p A)`.

Cabeza CLV: `z_u w + cb`, con normalización del target calculada en train. Cabeza recomendación: producto escalar `z_u · v_p / sqrt(d) + sesgo_p`, con BCE sobre positivos y negativos muestreados. La pérdida conjunta es MSE + lambda*BCE. La no linealidad tanh, embeddings y proyección de atributos se entrenan por gradiente. K-Means se aplica después a `z_u`; no existe una tercera cabeza supervisada de segmentación.

El motor de referencia es NumPy con Adam y backprop explícito. La implementación PyTorch usa ParameterDict para representar exactamente las mismas matrices y una prueba opcional compara sus salidas. El backend PyTorch no fue ejecutado durante la preparación. En v2 mueve modelo y tensores a CUDA si está disponible, con selección explícita auto/cuda/cpu; la validación sigue en NumPy.

## 5. Selección y evaluación

Early stopping utiliza la pérdida combinada de validación, con paciencia 20. La configuración principal es d=16, lambda=2, tres negativos y regularización 1e-4. Se seleccionó en validación buscando el mayor NDCG entre configuraciones con RMSE no superior al 115 % del mejor RMSE de la exploración. Este criterio de compromiso es una decisión de diseño exploratoria, no una hipótesis preregistrada. El estudio de sensibilidad es OAT. Las ablaciones usan lambda=1 en recomendación aislada; multiplicar toda su pérdida por una constante no cambia el objetivo aislado, aunque puede afectar la optimización. Incluye lambda, dimensión, negativos, regularización y target logarítmico. Para el target logarítmico se usa corrección de retransfomación estimada solo con residuos train; sus resultados pueden ser peores.

Los baselines CLV son media de gasto objetivo train, Ridge (regresión lineal regularizada), Random Forest y Gradient Boosting. La recomendación compara popularidad y SVD de la matriz de positivos train. Todos comparten cohorte, catálogo y ventanas.

MAE y RMSE se expresan en BRL; Spearman mide ordenación y el lift compara gasto medio observado del decil predicho superior con la media. Precision@10, Recall@10 y NDCG@10 se calculan sobre usuarios con al menos una compra relevante futura en el catálogo. Se informa `eligible_users`. Cobertura es la fracción de productos recomendados a esos usuarios. No se usan candidatos negativos muestreados para inflar la evaluación: el ranking considera los 300 productos.

K se explora de 2 a 6 con silueta en train; se calcula estabilidad ARI entre dos semillas. El PCA y los centroides se ajustan en train y se aplican a test. Las siluetas de espacios distintos no demuestran por sí solas utilidad comercial. Las acciones de marketing son propuestas, no resultados de campañas ejecutadas.

## 6. Limitaciones

El generador incorpora relaciones por diseño. Que un modelo las recupere solo demuestra que aprende ese escenario. La cohorte sintética tiene recurrencia elevada deliberadamente y no es una calibración realista de Olist. No debe utilizarse para estimar tasas reales de recompra, beneficios o eficacia publicitaria. La evaluación original usa la misma cohorte antigua, por lo que puede carecer de compras futuras en el catálogo reducido; no representa una evaluación exhaustiva de todo Olist.

El modelo es transductivo y no acepta clientes desconocidos como si tuviera un embedding entrenado. La API devuelve 404. El análisis de historial escaso no equivale a verdadero cold-start. La v2 añade una variante de recomendación por categorías mediante agregación de logits, clasificación independiente de recompra y evaluación de productos con pocas interacciones. No implementa una cabeza adicional de recompra ni calibración de afinidades como probabilidades de compra. No hay inferencia causal de campañas ni intervalos predictivos individuales.
