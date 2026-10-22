# Anexo v2 · Experimentos adicionales

Escenario semisintético de los informes actuales. No evidencia real de Olist. Motor de los pesos: torch.

## Versión por categorías

Media de logits de productos por categoría, sin entrenamiento independiente. Se compara con popularidad y se evalúa con K=3 sobre todas las categorías del catálogo reducido. No se interpreta como probabilidad.

| model                                   |   categories |   Precision@3 |   Recall@3 |   NDCG@3 |   coverage |   eligible_users |
|:----------------------------------------|-------------:|--------------:|-----------:|---------:|-----------:|-----------------:|
| Red conjunta · agregación por categoría |           10 |        0.3580 |     0.7766 |   0.7804 |     1.0000 |             1569 |
| Popularidad de categoría                |           10 |        0.1838 |     0.3651 |   0.2970 |     0.3000 |             1569 |

## Productos con pocas interacciones

Estratos por pares cliente-producto de train. Se restringen los positivos evaluados al estrato, manteniendo todos los productos como candidatos. Cero casos no permite una conclusión.

| group                    |   products |   relevant_pairs | model        |   Precision@10 |   Recall@10 |   NDCG@10 |   coverage |   eligible_users |
|:-------------------------|-----------:|-----------------:|:-------------|---------------:|------------:|----------:|-----------:|-----------------:|
| 0-2 interacciones train  |          3 |                3 | Red conjunta |         0.0333 |      0.3333 |    0.1003 |     0.0767 |                3 |
| 0-2 interacciones train  |          3 |                3 | Popularidad  |         0.0000 |      0.0000 |    0.0000 |     0.0333 |                3 |
| 3-10 interacciones train |         50 |              376 | Red conjunta |         0.0136 |      0.1039 |    0.0388 |     0.5867 |              308 |
| 3-10 interacciones train |         50 |              376 | Popularidad  |         0.0000 |      0.0000 |    0.0000 |     0.0333 |              308 |
| 11+ interacciones train  |        247 |             4400 | Red conjunta |         0.0851 |      0.2989 |    0.2080 |     0.7467 |             1523 |
| 11+ interacciones train  |        247 |             4400 | Popularidad  |         0.0255 |      0.0891 |    0.0550 |     0.0333 |             1523 |

## Target binario de recompra

Etiqueta gasto futuro>0. Regresión logística y árboles potenciados independientes frente a tasa train. No es una cabeza adicional de la red ni probabilidad de responder a publicidad.

| model                  |   customers |   positive_rate |   ROC_AUC |   PR_AUC_AP |   Brier |   LogLoss |
|:-----------------------|------------:|----------------:|----------:|------------:|--------:|----------:|
| Tasa de recompra train |        2500 |          0.6312 |    0.5000 |      0.6312 |  0.2545 |    0.7139 |
| Regresión logística    |        2500 |          0.6312 |    0.7317 |      0.8195 |  0.2562 |    0.7647 |
| Árboles potenciados    |        2500 |          0.6312 |    0.7279 |      0.8352 |  0.2582 |    0.7533 |

## Lectura responsable

Las comparaciones corresponden al escenario y los informes guardados. No se comparan directamente escalas de gasto, clasificación y ranking como si fueran la misma métrica. No se ha demostrado generalización a clientes o productos desconocidos. La interpretación comercial necesita una prueba adicional con datos válidos de campaña.
