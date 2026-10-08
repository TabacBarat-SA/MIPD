"""
Bloque 2 – Arquitectura MultiTaskNet.

Arquitectura:
    ┌─────────────────────────────────────────────────────────┐
    │                    MultiTaskNet                         │
    │                                                         │
    │  Embedding(cliente)  +  [features_cliente]              │
    │       ↓                                                 │
    │  [Tronco MLP compartido (opcional)]                     │
    │       ├──→ Cabeza CLV  →  clv_pred (escalar)            │
    │       │                                                 │
    │  Embedding(producto)                                    │
    │       ↓                                                 │
    │  [cat(tronco_out, emb_producto) → MLP rec]              │
    │       └──→ Cabeza Rec  →  rec_logit (escalar)           │
    └─────────────────────────────────────────────────────────┘

Decisiones:
    - Cabeza CLV: solo usa el embedding/representación del cliente.
    - Cabeza Rec: concatena la representación del cliente con el embedding
      del producto y pasa por un MLP → logit.
    - Embeddings inicializados con Normal(0, 0.01).
    - LayerNorm y Dropout configurables.
    - ``usar_features=True``: concatena features numéricas al embedding
      antes del tronco. Requiere que el caller normalice las features.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _bloque_mlp(
    dim_entrada: int,
    capas: list[int],
    dropout: float,
    usar_layer_norm: bool,
) -> nn.Sequential:
    """Crea un MLP con ReLU, Dropout y LayerNorm (opcional).

    Parámetros
    ----------
    dim_entrada : int
        Dimensión de entrada al primer bloque.
    capas : list[int]
        Dimensiones de las capas ocultas/salida.
    dropout : float
        Probabilidad de Dropout (0 = sin dropout).
    usar_layer_norm : bool
        Si True, aplica LayerNorm después de cada Linear.
    """
    bloques: list[nn.Module] = []
    d_in = dim_entrada
    for d_out in capas:
        bloques.append(nn.Linear(d_in, d_out))
        if usar_layer_norm:
            bloques.append(nn.LayerNorm(d_out))
        bloques.append(nn.ReLU())
        if dropout > 0:
            bloques.append(nn.Dropout(dropout))
        d_in = d_out
    return nn.Sequential(*bloques)


class MultiTaskNet(nn.Module):
    """Red neuronal multi-tarea: CLV + recomendación.

    Parámetros
    ----------
    n_clientes : int
        Número total de clientes (tamaño del embedding).
    n_productos : int
        Número total de productos (tamaño del embedding).
    dim_embedding : int
        Dimensión de los embeddings de cliente y producto.
    capas_tronco : list[int]
        Capas del MLP compartido. Lista vacía = sin tronco.
    dropout : float
        Probabilidad de Dropout.
    usar_layer_norm : bool
        Si True, usa LayerNorm en los MLPs.
    usar_features : bool
        Si True, concatena features numéricas de cliente al embedding.
    n_features_cliente : int
        Número de features numéricas de cliente (ignorado si usar_features=False).
    """

    def __init__(
        self,
        n_clientes: int,
        n_productos: int,
        dim_embedding: int = 64,
        capas_tronco: list[int] | None = None,
        dropout: float = 0.3,
        usar_layer_norm: bool = True,
        usar_features: bool = True,
        n_features_cliente: int = 9,
    ) -> None:
        super().__init__()

        if capas_tronco is None:
            capas_tronco = [128, 64]

        self.n_clientes = n_clientes
        self.n_productos = n_productos
        self.dim_embedding = dim_embedding
        self.usar_features = usar_features
        self.n_features_cliente = n_features_cliente if usar_features else 0

        # ── Embeddings ────────────────────────────────────────────────────── #
        self.emb_cliente = nn.Embedding(n_clientes, dim_embedding)
        self.emb_producto = nn.Embedding(n_productos, dim_embedding)

        # Inicialización con Normal pequeña para estabilidad
        nn.init.normal_(self.emb_cliente.weight, std=0.01)
        nn.init.normal_(self.emb_producto.weight, std=0.01)

        # ── Dimensiones ───────────────────────────────────────────────────── #
        dim_cliente_raw = dim_embedding + self.n_features_cliente  # tras concat features

        # ── Tronco compartido (MLP) ───────────────────────────────────────── #
        if capas_tronco:
            self.tronco = _bloque_mlp(
                dim_entrada=dim_cliente_raw,
                capas=capas_tronco,
                dropout=dropout,
                usar_layer_norm=usar_layer_norm,
            )
            dim_tronco_out = capas_tronco[-1]
        else:
            self.tronco = nn.Identity()
            dim_tronco_out = dim_cliente_raw

        # ── Cabeza CLV ────────────────────────────────────────────────────── #
        # Entrada: representación del cliente (salida del tronco)
        self.cabeza_clv = nn.Linear(dim_tronco_out, 1)
        nn.init.zeros_(self.cabeza_clv.bias)

        # ── Cabeza Recomendación ──────────────────────────────────────────── #
        # Entrada: [rep_cliente || emb_producto] → MLP → logit
        dim_rec_in = dim_tronco_out + dim_embedding
        self.cabeza_rec = nn.Sequential(
            nn.Linear(dim_rec_in, 64),
            nn.LayerNorm(64) if usar_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        cliente: torch.Tensor,
        producto: torch.Tensor,
        features_cliente: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pasa cliente y producto por la red.

        Parámetros
        ----------
        cliente : Tensor[long] shape (B,)
            Índices de clientes.
        producto : Tensor[long] shape (B,)
            Índices de productos.
        features_cliente : Tensor[float] shape (B, n_features_cliente) | None
            Features numéricas del cliente. Obligatorio si ``usar_features=True``.

        Devuelve
        -------
        clv_pred : Tensor[float] shape (B,)
            Predicción de CLV (en escala log1p).
        rec_logit : Tensor[float] shape (B,)
            Logit de probabilidad de compra (sin sigmoide).
        """
        # Embeddings base
        e_c = self.emb_cliente(cliente)       # (B, dim_embedding)
        e_p = self.emb_producto(producto)     # (B, dim_embedding)

        # Concatenar features de cliente si procede
        if self.usar_features:
            if features_cliente is None:
                features_cliente = torch.zeros(e_c.size(0), self.n_features_cliente, device=e_c.device, dtype=e_c.dtype)
            x_c = torch.cat([e_c, features_cliente], dim=-1)  # (B, dim+n_feat)
        else:
            x_c = e_c  # (B, dim_embedding)

        # Tronco compartido
        rep_c = self.tronco(x_c)  # (B, dim_tronco_out)

        # Cabeza CLV
        clv_pred = self.cabeza_clv(rep_c).squeeze(-1)  # (B,)

        # Cabeza Recomendación
        rec_in = torch.cat([rep_c, e_p], dim=-1)       # (B, dim_tronco_out + dim_emb)
        rec_logit = self.cabeza_rec(rec_in).squeeze(-1)  # (B,)

        return clv_pred, rec_logit

    # ------------------------------------------------------------------ #
    #  Métodos auxiliares                                                  #
    # ------------------------------------------------------------------ #

    def get_customer_embeddings(self) -> torch.Tensor:
        """Devuelve la matriz de embeddings de clientes (n_clientes, dim_embedding)."""
        return self.emb_cliente.weight.detach().cpu()

    def get_product_embeddings(self) -> torch.Tensor:
        """Devuelve la matriz de embeddings de productos (n_productos, dim_embedding)."""
        return self.emb_producto.weight.detach().cpu()

    @torch.no_grad()
    def score_all_products(
        self,
        customer_idx: int,
        device: torch.device | None = None,
        features_cliente: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Puntúa todos los productos para un cliente dado.

        Parámetros
        ----------
        customer_idx : int
            Índice del cliente.
        device : torch.device | None
            Dispositivo de cómputo. Si None, usa CPU.
        features_cliente : Tensor[float] shape (n_features_cliente,) | None
            Features del cliente, requerido si ``usar_features=True``.

        Devuelve
        -------
        scores : Tensor[float] shape (n_productos,)
            Logits de compra para cada producto.
        """
        if device is None:
            device = next(self.parameters()).device

        n = self.n_productos
        c_tensor = torch.full((n,), customer_idx, dtype=torch.long, device=device)
        p_tensor = torch.arange(n, dtype=torch.long, device=device)

        feat = None
        if self.usar_features and features_cliente is not None:
            feat = features_cliente.to(device).unsqueeze(0).expand(n, -1)

        _, logits = self.forward(c_tensor, p_tensor, feat)
        return logits.cpu()

    @classmethod
    def desde_config(cls, cfg: dict[str, Any]) -> "MultiTaskNet":
        """Construye la red a partir del dict de configuración."""
        m = cfg["modelo"]
        return cls(
            n_clientes=m["n_clientes"],
            n_productos=m["n_productos"],
            dim_embedding=m["dim_embedding"],
            capas_tronco=m.get("capas_tronco", [128, 64]),
            dropout=m["dropout"],
            usar_layer_norm=m["usar_layer_norm"],
            usar_features=m["usar_features"],
            n_features_cliente=m.get("n_features_cliente", 9),
        )
