"""Diagnostico sin entrenar ni modificar artefactos."""
from pathlib import Path
import sys,importlib.util
root=Path(__file__).resolve().parents[1]
print('Python:',sys.version.split()[0]);print('Proyecto:',root)
for relative in ['.streamlit/config.toml','data/processed/synthetic.npz','artifacts/synthetic_multi.npz','reports/synthetic_neural_clv.csv']:
    print(('OK ' if (root/relative).exists() else 'FALTA ')+relative)
if importlib.util.find_spec('torch') is None:
    print('PyTorch no instalado en este interprete. Usa uv sync y ejecuta con uv run.');sys.exit(1)
import torch
print('PyTorch:',torch.__version__);print('CUDA del paquete:',torch.version.cuda);print('CUDA disponible:',torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:',torch.cuda.get_device_name(0));x=torch.ones(3,device='cuda');print('Operacion CUDA:',(x+x).cpu().tolist())
else:print('Se puede abrir el panel y entrenar en CPU. No se afirma que se este usando CUDA.')
