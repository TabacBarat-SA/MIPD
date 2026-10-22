# Estado de comprobaciones · v2

## Ejecutado en el entorno de preparación

- Suite local: **16 pruebas superadas y 3 omitidas**. Las omitidas son equivalencia PyTorch, entrenador PyTorch CPU y entrenador CUDA, por ausencia de PyTorch/GPU en este entorno.
- Integridad: claves, pagos, fechas, conservación de originales y prueba anti-fuga.
- Negocio: cobertura de grupos, alias únicos, filtros de audiencia, ordenación, coste supuesto, selección vacía, categorías históricas y coherencia de predicciones cacheadas.
- Experimentos adicionales: informes de categorías, productos con pocas interacciones y recompra calculados para ambos escenarios.
- AppTest: siete páginas en ambos escenarios, sin excepciones ni mensajes de error; se comprueba también audiencia vacía.
- Chromium: siete páginas con configuración clara y siete con configuración oscura, sin excepciones. Descarga de audiencia de 100 filas comprobada en ambas configuraciones. No se afirma cobertura de todos los navegadores o preferencias guardadas.
- Vista móvil: guía a 390 px sin desbordamiento horizontal del documento en ambas configuraciones. No es una auditoría móvil exhaustiva de todas las páginas.
- API: contratos incluidos en la suite. Las consultas de pasarela se habían comprobado en la versión base; Docker no se ha construido aquí.
- Sintaxis: código Python compilado y configuración TOML validada.

## Pendiente de verificar en el equipo destino

- Instalación completa con uv: el acceso a índices devolvió 403 en este entorno. No se entrega un lock ficticio.
- Entrenamiento PyTorch/CUDA y comparación de sus resultados.
- Construcción Docker y recorrido REST completo del despliegue actualizado.
- Revisión de usabilidad con personas de marketing no técnicas. Las pruebas de ejecución no demuestran por sí solas comprensión de usuario.

Los pesos de demostración siguen siendo NumPy. El parche no incluye pesos ni datos ni informes calculados, para no sobrescribir una ejecución local. Tras aplicarlo se deben regenerar los informes adicionales y la memoria con los pesos del usuario.
