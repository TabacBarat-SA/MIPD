"""Backend PyTorch con la MISMA arquitectura y pesos NPZ que NumPy.
Instalacion: uv sync. No es necesario para abrir la demo.
"""
import numpy as np
import torch
from torch import nn

class Cliente360Net(nn.Module):
    def __init__(self,reference):
        super().__init__()
        self.weights=nn.ParameterDict({k:nn.Parameter(torch.tensor(v.copy())) for k,v in reference.p.items()})
        self.register_buffer('categories',torch.tensor(reference.categories,dtype=torch.long))
        self.register_buffer('attrs',torch.tensor(reference.attrs,dtype=torch.float32))
        self.dim=reference.dim
    def encode(self,x):
        p=self.weights
        return torch.tanh(p['U']+x@p['W']+p['b'])
    def forward(self,x,pairs):
        p=self.weights;z=self.encode(x)
        v=torch.tanh(p['V']+p['C'][self.categories]+self.attrs@p['A'])
        u,pr=pairs.T
        return z@p['w']+p['cb'][0],(z[u]*v[pr]).sum(1)/np.sqrt(self.dim)+p['pb'][pr]

class TorchTrainer:
    """Auto usa CUDA si existe. CLIENTE360_DEVICE=cuda exige GPU; cpu fuerza CPU."""
    def __init__(self, reference, lr=.01, decay=1e-4):
        import os
        torch.manual_seed(42)
        requested = os.environ.get('CLIENTE360_DEVICE', 'auto').lower()
        if requested not in ['auto', 'cuda', 'cpu']:
            raise ValueError('CLIENTE360_DEVICE debe ser auto, cuda o cpu.')
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Se solicito CUDA, pero PyTorch no detecta GPU. Ejecuta scripts/comprobar_entorno.py.')
        self.device = torch.device('cuda:0' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(2)
        self.net = Cliente360Net(reference).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=decay)
        self.cached_tensors = None
        self.epoch = 0
        print('PyTorch | dispositivo:', self.device, '|', torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU (sin aceleracion CUDA)', flush=True)

    def update(self, x, y, pairs, labels, rw, cw):
        if self.cached_tensors is None:
            self.cached_tensors = (
                torch.as_tensor(x, dtype=torch.float32, device=self.device),
                torch.as_tensor(y, dtype=torch.float32, device=self.device),
                torch.as_tensor(pairs, dtype=torch.long, device=self.device),
                torch.as_tensor(labels, dtype=torch.float32, device=self.device),
            )
        x_gpu, y_gpu, pairs_gpu, labels_gpu = self.cached_tensors
        self.net.train()
        self.optimizer.zero_grad(set_to_none=True)
        clv, rec = self.net(x_gpu, pairs_gpu)
        cl = nn.functional.mse_loss(clv, y_gpu)
        rl = nn.functional.binary_cross_entropy_with_logits(rec, labels_gpu) if labels_gpu.numel() else rec.sum() * 0
        (cw * cl + rw * rl).backward()
        nn.utils.clip_grad_value_(self.net.parameters(), 5.)
        self.optimizer.step()
        self.epoch += 1
        values = cl.detach().item(), rl.detach().item()
        if self.epoch == 1 or self.epoch % 10 == 0:
            print(f'Epoca {self.epoch}: CLV={values[0]:.4f}, recomendacion={values[1]:.4f}', flush=True)
        return values

    def export(self, reference):
        reference.p = {key: value.detach().cpu().numpy().copy() for key, value in self.net.weights.items()}
