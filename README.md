# 360 Analytics · Decisiones de marketing

Herramienta académica de Marketing Digital y Publicidad Interactiva para responder tres preguntas: **a quién prestar atención, qué productos ofrecer y cómo comprobar la propuesta**. Integra previsión de gasto a tres meses, recomendaciones y grupos de clientes.

> Demostración histórica con Olist. El escenario semisintético añade compras artificiales identificadas. No demuestra ventas reales, ROI ni eficacia publicitaria. No envía mensajes ni publica anuncios.

## 1. Si ya tenías la versión anterior

**No sobrescribas tus pesos entrenados con los de otra ejecución.**

- Para conservar tu entrenamiento: utiliza `Cliente360_actualizacion_v2.zip`. Extrae su contenido directamente en la raíz actual, donde está `pyproject.toml`, y acepta sustituir el código. El parche no contiene `data/`, `artifacts/` ni `reports/`.
- Para empezar en una carpeta nueva: utiliza `Cliente360_completo_v2.zip`. Incluye los datos y pesos de demostración originales, cuyo motor registrado es NumPy. No se presentan como pesos entrenados con CUDA.
- El código de PyTorch/CUDA y el nuevo `pyproject.toml` **ya están modificados** en ambos paquetes. No hace falta copiar manualmente la clase del README antiguo.
- La interfaz es ahora una aplicación web (carpeta `web/`). Consulta el apartado 3 para compilarla y abrirla. Ya no se usa Streamlit.

Después de aplicar el parche, ejecuta:

```bash
uv sync
uv run python -m cliente360.cli extended
uv run python -m cliente360.cli extended --source original
uv run python scripts/generar_memoria.py
uv run uvicorn services.web:app --port 8501
```

Los experimentos `extended` calculan informes adicionales sobre tus pesos actuales. No reentrenan ni cambian la red principal. Si has cambiado pesos y todavía no has recalculado `evaluate` y `segment`, sigue antes la secuencia completa del apartado 4.

## 2. Instalar por primera vez

### Requisitos

- Carpeta completa del proyecto o parche aplicado sobre una copia completa.
- Internet para instalar dependencias.
- Python 3.12, que puede instalar uv.
- GPU NVIDIA solo si quieres aceleración CUDA. No hace falta GPU para abrir la web.

### Windows: instalar uv

En PowerShell o Git Bash:

```bash
winget install --id=astral-sh.uv -e
```

Cierra y vuelve a abrir la terminal y comprueba:

```bash
uv --version
```

Para otros métodos y sistemas, consulta la [instalación oficial de uv](https://docs.astral.sh/uv/getting-started/installation/).

uv se instala en el ordenador, no como una carpeta del proyecto. **No ejecutes `uv init`**: el proyecto ya tiene configuración.

### Abrir la carpeta correcta

Abre una terminal en la carpeta donde están `pyproject.toml`, `README.md`, `web/` y `cliente360/`. No dentro de la subcarpeta que contiene los módulos Python.

```bash
uv python install 3.12
uv sync
```

Si el equipo ya comparte un `uv.lock` actualizado, utiliza `uv sync --locked`. No se incluye un lock inventado: la resolución de índices no pudo completarse en el entorno de preparación. La primera resolución válida del equipo debe guardarse y compartirse.

uv crea `.venv`. No hace falta activarlo ni instalar paquetes manualmente con pip. Todos los comandos de esta guía se ejecutan desde la raíz con `uv run`.

## 3. Abrir la web sin entrenar

La interfaz de 360 Analytics es una aplicación React (Vite, Tailwind, Motion y Recharts) que consulta una API FastAPI (`services/web.py`) con los resultados guardados. Necesitas Node 20 o superior para compilarla.

```bash
cd web
npm install
npm run build
cd ..
uv run uvicorn services.web:app --port 8501
```

Abre la [web local](http://localhost:8501). Detén el servidor con `Ctrl+C`.

Para desarrollar con recarga en caliente, deja la API en marcha y, en otra terminal, ejecuta `npm run dev` dentro de `web/` (usa http://localhost:5173 y redirige `/api` al puerto 8501).

Los pesos y resultados guardados permiten usar la aplicación sin reentrenar. **No ejecutes `all` para limitarte a abrir la demo.**

### Moneda

Los datos están en reales (BRL). La web los muestra en dólares con el tipo de cambio actual (`forex-python`, se consulta como máximo una vez por hora) y permite alternar a BRL desde el menú lateral. Sin conexión o sin tipo de cambio disponible se muestran reales.

### Tema claro y oscuro

La web respeta la preferencia del sistema y recuerda la elección del botón de tema. Los textos de las tablas, gráficos y avisos usan variables de color que cambian con el tema, así que no hay texto claro sobre fondo claro.

## 4. Entrenar con PyTorch y CUDA

### Configuración incluida

`pyproject.toml` incluye PyTorch 2.9.1 y el índice CUDA 12.8 para Windows/Linux. Es una combinación publicada oficialmente, no una afirmación de que sea la más reciente. En macOS no se habilita CUDA. PyTorch es una dependencia normal: **no utilices `--extra torch`**.

`TorchTrainer` mueve modelo y tensores al dispositivo seleccionado y reutiliza los tensores de entrenamiento. El modo predeterminado es automático: CUDA si está disponible; en caso contrario, CPU con aviso explícito.

### Comprobar el equipo

```bash
nvidia-smi
uv run python scripts/comprobar_entorno.py
```

Para CUDA, la comprobación debe indicar `CUDA disponible: True`, identificar la GPU y completar una suma de tensores en GPU. Que funcione `nvidia-smi` por sí solo no demuestra que PyTorch esté utilizando CUDA.

### Entrenar y actualizar la web

Detén el servidor web y guarda una copia de `artifacts/`, `reports/` y `data/processed/` si quieres conservar la ejecución anterior. Ejecuta uno por uno, continuando solo si el anterior termina sin errores:

```bash
uv run python -m cliente360.cli train --backend torch --epochs 100
uv run python -m cliente360.cli evaluate
uv run python -m cliente360.cli segment
uv run python -m cliente360.cli extended
uv run uvicorn services.web:app --port 8501
```

Al comenzar, el entrenador muestra `PyTorch | dispositivo: cuda:0` o `cpu`. El registro `artifacts/synthetic_training.json` conserva backend y dispositivo para los nuevos entrenamientos.

`train` ejecuta tres modelos: red conjunta, solo gasto y solo recomendación. `--epochs 100` es un máximo por modelo; early stopping puede detenerlo antes. Para una comprobación funcional puedes usar 30 épocas, pero no presupongas la misma calidad.

### Exigir CUDA y evitar una ejecución silenciosa en CPU

Git Bash o Linux:

```bash
CLIENTE360_DEVICE=cuda uv run python -m cliente360.cli train --backend torch --epochs 100
```

PowerShell:

```powershell
$env:CLIENTE360_DEVICE = "cuda"
uv run python -m cliente360.cli train --backend torch --epochs 100
Remove-Item Env:CLIENTE360_DEVICE
```

Con ese ajuste, si CUDA no está disponible se produce un error claro. Para forzar CPU usa `CLIENTE360_DEVICE=cpu` con la sintaxis de tu terminal.

### Alternativa sin PyTorch en el cálculo

```bash
uv run python -m cliente360.cli train --backend numpy --epochs 100
uv run python -m cliente360.cli evaluate
uv run python -m cliente360.cli segment
uv run python -m cliente360.cli extended
```

La instalación del proyecto sigue incluyendo PyTorch, pero esta ruta entrena con el motor NumPy de referencia.

### Rendimiento y memoria

No hay una duración garantizada. El entrenamiento es full-batch y la validación sigue en NumPy/CPU; los pesos se exportan después de cada época. Baselines y segmentación tampoco pasan a GPU. Cierra aplicaciones que ocupen VRAM si necesitas liberar memoria. Reducir las épocas no reduce el tamaño de los tensores de cada paso.

## 5. Usar la herramienta sin conocimientos técnicos

### Recorrido de negocio

1. **Resumen del negocio:** tamaño de la muestra, gasto previsto y distribución por grupos.
2. **Entender los grupos:** qué observamos, qué acción probar, canal propuesto e indicador.
3. **Conocer a un cliente:** historial, gasto previsto, grupo y selección de productos.
4. **Preparar una campaña:** objetivo, grupos, filtros, límite de clientes y coste supuesto de contacto. Permite descargar una audiencia de demostración.
5. **Comprobar los resultados:** compara la red con métodos más sencillos. Los experimentos técnicos están separados de las decisiones de negocio.
6. **Cómo funciona:** relación entre datos, modelo y decisiones, con límites explícitos.
7. **Guía de uso:** recorrido y diccionario de conceptos.

### Cómo leer las cifras

- **Gasto previsto / CLV:** gasto estimado en los próximos tres meses. No beneficio, valor vitalicio ni ingreso atribuible a una campaña.
- **Última compra:** días transcurridos en el corte histórico del 01/04/2018, no desde hoy.
- **Grupo:** clientes con patrones similares. Los nombres comerciales son reglas descriptivas sobre sus medias; no clases entrenadas de abandono o fidelidad.
- **Afinidad:** puntuación que ordena productos, no porcentaje de probabilidad de compra.
- **BRL / R$:** reales brasileños. No se convierten los datos a euros.
- **Datos simulados:** compras artificiales identificadas, no resultados reales de Olist.

Los alias `Cliente 0001` y las referencias cortas de producto son ayudas de interfaz. Los IDs completos se conservan en las descargas. No se inventan nombres comerciales, imágenes de productos ni identidades de clientes.

### Campañas: alcance real

El módulo prepara una audiencia histórica de demostración, ordenada por gasto previsto. No dispone de emails, consentimiento, clics o exposición publicitaria. No envía mensajes ni activa anuncios.

El coste mostrado es `clientes seleccionados × coste supuesto por contacto`, introducido por el usuario. No se calcula ROI ni aumento de ventas. El gasto previsto no se atribuye a la campaña. Antes de descargar se pide reconocer estos límites.

El objetivo de campaña se guarda como contexto; no cambia automáticamente la puntuación ni optimiza una política de tratamiento. El usuario elige grupos y filtros. Las propuestas de canal y KPI son hipótesis de trabajo, no recomendaciones de eficacia demostrada.

## 6. Reproducir todo el experimento

Solo si quieres regenerar datos y resultados. Conserva una copia antes.

```bash
uv run python -m cliente360.cli all --backend torch --epochs 100 --customers 2500
uv run python -m cliente360.cli extended
uv run python -m cliente360.cli experiments --backend torch --epochs 100
```

Para NumPy sustituye `--backend torch` por `--backend numpy`. El CLI sigue usando NumPy si no especificas backend.

### Datos originales: misma cohorte y catálogo

Después de generar la cohorte:

```bash
uv run python -m cliente360.cli prepare --source original
uv run python -m cliente360.cli train --source original --backend torch --epochs 100
uv run python -m cliente360.cli evaluate --source original
uv run python -m cliente360.cli segment --source original
uv run python -m cliente360.cli extended --source original
```

No uses `all --source original` para evitar la generación: `all` también llama al generador. Si cambias la cohorte, regenera ambos escenarios. No mezcles índices y artefactos de muestras distintas.

### Qué hace cada comando

- `generate`: conserva originales y crea la ampliación y su trazabilidad.
- `prepare`: construye snapshots, índices y tensores.
- `train`: pesos, curvas y métricas de la red conjunta y ablaciones.
- `evaluate`: baselines, selección retrospectiva y análisis de historial escaso de clientes.
- `segment`: embeddings, PCA, K-Means y perfiles.
- `extended`: evaluación por categorías, estratos de productos con pocas interacciones y target binario de recompra.
- `experiments`: sensibilidad OAT de configuración en validación; no cambia automáticamente el modelo principal.

La versión por categoría promedia logits de productos; no entrena otra red. La recompra se estudia con clasificadores independientes y tasa base, no con una cabeza adicional multi-tarea. Las métricas de recompra no se presentan como probabilidad de responder a publicidad.

## 7. Compartir el proyecto y conservar resultados

Comparte código, `web/` (sin `node_modules/` ni `dist/`), `pyproject.toml`, `.python-version`, el lock válido cuando exista, datos, artefactos e informes de la misma ejecución. No compartas `.venv/`.

**El `.gitignore` inicial excluye `data/` y `artifacts/`.** Clonar solo el repositorio no garantiza recibir los datos o pesos. Distribuye esos directorios mediante el paquete completo o un mecanismo acordado.

El parche v2 conserva tus pesos porque no los incluye. Tras aplicarlo, recalcula `extended` con tus pesos. Si vuelves a entrenar, recalcula también `evaluate` y `segment`. Reinicia el servidor y los servicios para vaciar cachés; el botón «Actualizar datos» de la web limpia la caché del navegador, no las de procesos REST remotos.

## 8. Pruebas, memoria y presentación

```bash
uv run pytest -q
```

Las pruebas de PyTorch requieren esa dependencia. Las pruebas de CUDA solo pueden comprobarse en un equipo con GPU compatible. No se ha ejecutado CUDA en el entorno de preparación.

Para regenerar tablas y gráficos de la memoria base:

```bash
uv run python scripts/generar_memoria.py
```

El generador v2 lee los informes actuales, registra el backend y actualiza memoria, anexo y gráficos. Antes de ejecutarlo, recalcula `evaluate`, `segment` y `extended` con los mismos pesos. Revisa siempre la interpretación y la correspondencia de archivos antes de presentar; generar texto no verifica por sí solo la coherencia de una ejecución.

La memoria y el guion están en Markdown. No se incluye una presentación PowerPoint terminada ni se garantiza una calificación. `docs/cobertura.md` distingue alcance, evidencia y pendientes frente a la propuesta.

## 9. Servicios REST y Docker

```bash
docker compose up --build
```

[Web](http://localhost:8501) · [Pasarela](http://localhost:8000/health)

Cinco servicios: datos (8001 interno), modelo (8002), segmentación (8003), evaluación (8004) y pasarela (8000) y web (8501). Las API consultan artefactos; el entrenamiento y los cálculos se realizan por CLI. El panel utiliza REST para las consultas principales y el volumen común para las vistas derivadas de negocio y los informes adicionales.

Los volúmenes se montan en solo lectura. No se reentrena al arrancar. El Dockerfile compila la web en una etapa Node y copia `web/dist`. Compose no configura GPU para entrenamiento: la ruta CUDA descrita aquí es local.

```bash
docker compose down
```

La construcción Docker sigue sin verificarse en el entorno de preparación. No es un despliegue productivo con autenticación, permisos o monitorización.

## 10. Problemas frecuentes

- **uv no se reconoce:** reinicia la terminal y comprueba `uv --version`.
- **No encuentra módulos o pyproject:** abre la raíz correcta y utiliza `uv run`.
- **Extra torch no definido:** elimina `--extra torch`; ahora es dependencia normal.
- **CUDA no disponible:** ejecuta el diagnóstico con `uv run`, comprueba el índice `pytorch-cu128` y vuelve a sincronizar. No basta con usar otro Python que sí tenga CUDA.
- **Error de VRAM:** libera memoria o usa CPU/NumPy. El entrenamiento no implementa minibatches.
- **Faltan datos o pesos:** recupera el paquete completo o reconstruye desde los CSV originales. No crees archivos vacíos.
- **Resultados antiguos:** recalcula informes y segmentos, reinicia servicios y pulsa «Actualizar datos» en la web.
- **Sin clientes en la campaña:** amplía grupos o reduce filtros. Una audiencia vacía se muestra sin inventar clientes.
- **Ceros en recomendación original:** revisa `eligible_users`. Cero casos significa ausencia de evaluación, no evidencia de calidad.
- **Errores de red al instalar:** comprueba acceso a PyPI y al índice PyTorch. No desactives TLS como solución general.

## 11. Datos y protocolo

Identidad persistente: `customer_unique_id`. Se crea un `customer_id` nuevo por pedido artificial. Los nueve CSV originales se conservan sin sobrescribir en `data/raw/`.

La ejecución de demostración seleccionó 2.500 personas y 300 productos conocidos antes de abril de 2017 y añadió 30.936 pedidos. Consulta `reports/synthetic_audit.json` para las cifras de tu ejecución. Los perfiles latentes del generador se guardan para auditoría, nunca como features.

- Train: corte 01/10/2017; etiquetas hasta 01/01/2018.
- Validación: corte 01/01/2018; etiquetas hasta 01/04/2018.
- Test: corte 01/04/2018; etiquetas hasta 01/07/2018.

Las entradas son anteriores al corte. Las etiquetas usan inicio incluido y final excluido. Los mismos clientes aparecen en cortes distintos: no se demuestra generalización a clientes desconocidos.

## 12. Estructura y fuentes

```text
web/              Interfaz web (React + Vite + Tailwind)
cliente360/       Datos, redes, negocio, entrenamiento y evaluación
services/         Servicios REST, pasarela y API de la web (web.py)
data/             Originales, ampliación y datos procesados
artifacts/        Pesos y embeddings
reports/          Métricas y auditoría
docs/             Memoria, anexo v2, metodología y cobertura
scripts/          Diagnóstico y generación de memoria
notebooks/        Exploración
tests/            Pruebas de integridad, negocio y contratos
```

Referencias: [uv y PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/), [versiones PyTorch](https://pytorch.org/get-started/previous-versions/).

Fuente de datos: CSV de Brazilian E-Commerce Public Dataset by Olist aportados para el proyecto. Sus condiciones y autoría corresponden a su fuente. La ampliación se identifica en `provenance.csv`; no se asigna una licencia nueva a los originales.
