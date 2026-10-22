# Cobertura v2 frente a MIPD (7).pdf

La propuesta pide un producto utilizable por una persona no técnica y una evaluación que permita discutir si el Deep Learning mejora o no las decisiones de marketing. Una interfaz pulida no sustituye los experimentos ni garantiza una nota.

| Elemento de la propuesta | Entrega v2 | Evidencia o límite |
|---|---|---|
| Cinco tablas y cliente persistente | Implementado | Agregación de pagos, customer_unique_id, claves y cronología probadas. |
| CLV, recomendación y segmentación compartidas | Implementado | Dos cabezas y K-Means posterior sobre embeddings. |
| PyTorch | Código incluido | Pesos de demo proceden de NumPy. Entrenamiento PyTorch pendiente de ejecutar en destino. |
| CUDA | Configuración y código incluidos | Modelo y tensores al dispositivo, caché de tensores, selección auto/cuda/cpu y registro. Sin GPU disponible para verificar aquí. |
| Gasto bruto y logaritmo | Implementado en sensibilidad | Estudio OAT de validación, no búsqueda exhaustiva. |
| Probabilidad de recompra alternativa | Comparación añadida | Target gasto>0 con clasificadores independientes y tasa base. No cabeza adicional de la red. |
| Recomendación por categorías | Variante evaluada | Media de logits de producto por categoría frente a popularidad; no red independiente. |
| Modelos clásicos CLV y recomendación | Implementado | Media, Ridge, RF, boosting, popularidad y SVD. |
| Ablación | Implementado | Red conjunta frente a CLV y recomendación aisladas. |
| Sensibilidad lambda, dimensión, negativos y regularización | Implementado | Resultados de validación; no se afirma óptimo global. |
| Clientes con poco historial | Implementado | Estratos de historial, no clientes nuevos desconocidos. |
| Productos con poco historial | Añadido | Estratos por interacciones train y evaluación de positivos del estrato con catálogo completo. |
| Segmentación RFM frente a embeddings | Implementado | Silueta, estabilidad ARI, PCA y perfiles. Utilidad comercial no demostrada por silueta. |
| Perfilado con categorías preferidas por grupo | Añadido | Tres categorías con más artículos antes del corte y su proporción; no se usa compra futura. |
| Simulación de impacto de negocio | Retrospectiva | Selección top CLV frente a azar; no ROI ni efecto causal. |
| Panel de negocio | Rediseñado | Siete páginas, guía, glosario, acciones, canales, KPI y audiencia descargable. |
| Cinco servicios REST y volumen común | Incluidos | Cálculos por CLI; API de consulta. Panel combina REST y vistas del volumen. |
| Docker en una orden | Configuración incluida | Construcción y despliegue no verificados aquí. No GPU en Compose. |
| Memoria y presentación | Parcial | Memoria y anexo Markdown; guion, no PowerPoint final. |
| Datos originales Olist | Conservados | Ampliación semisintética explícita. Resultados simulados no sustituyen validación real. |

## Antes de la entrega académica

1. Ejecutar PyTorch/CUDA en el equipo del grupo y conservar versiones, dispositivo, curvas y métricas.
2. Recalcular evaluación, segmentación y experimentos extendidos con esos mismos pesos.
3. Ejecutar Docker y probar el recorrido REST completo en destino.
4. Revisar memoria y anexo: eliminar conclusiones que no correspondan a la ejecución presentada.
5. Revisar la interpretación de los perfiles y preparar la presentación final.
6. Si se exige una cabeza de recompra multi-tarea o un recomendador de categoría entrenado aparte, implementarlos y evaluarlos; no confundir las variantes actuales con esas arquitecturas.
7. Documentar claramente qué preguntas se responden con datos reales y cuáles solo con el escenario semisintético.
