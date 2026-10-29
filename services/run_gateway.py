"""Ejecuta pasarela y web en el quinto contenedor y propaga su terminacion."""
import os,subprocess,sys,time
os.environ.setdefault('API_URL','http://127.0.0.1:8000')
commands=[[sys.executable,'-m','uvicorn','services.gateway:app','--host','0.0.0.0','--port','8000'],[sys.executable,'-m','uvicorn','services.web:app','--host','0.0.0.0','--port','8501']]
children=[subprocess.Popen(cmd) for cmd in commands]
try:
    while all(p.poll() is None for p in children):time.sleep(.5)
finally:
    for p in children:p.terminate()
    for p in children:p.wait(timeout=10)
