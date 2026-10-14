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
      del producto y pasa por un MLP → logit (capas configurables en config.yaml).
    - Embeddings inicializados con Normal(0, 0.01).
    - LayerNorm y Dropout configurables.
    - ``usar_features=True``: concatena features numéricas al embedding
      antes del tronco. Requiere que el caller normalice las features.
    - ``predict_clv()``: método que infiere CLV sin necesitar un producto,
      evitando el anti-patrón de pasar un producto ficticio desde el pipeline.
    - ``score_todos_los_productos()``: función pública centralizada de scoring
      usada por evaluación y exportación para evitar duplicación de lógica.
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
    capas_rec : list[int]
        Capas del MLP de la cabeza de recomendación (sin contar la capa final →1).
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
        capas_rec: list[int] | None = None,
        dropout: float = 0.3,
        usar_layer_norm: bool = True,
        usar_features: bool = True,
        n_features_cliente: int = 9,
    ) -> None:
        super().__init__()

        if capas_tronco is None:
            capas_tronco = [128, 64]
        if capas_rec is None:
            capas_rec = [64]

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
        # Entrada: [rep_cliente || emb_producto] → MLP (capas configurables) → logit
        dim_rec_in = dim_tronco_out + dim_embedding
        # Construye MLP + capa de salida final
        capas_rec_completas = list(capas_rec) + [1]
        self.cabeza_rec = _bloque_mlp(
            dim_entrada=dim_rec_in,
            capas=capas_rec_completas[:-1],  # capas ocultas
            dropout=dropout,
            usar_layer_norm=usar_layer_norm,
        )
        # Última capa lineal sin activación ni dropout (logit crudo)
        self.cabeza_rec_out = nn.Linear(
            capas_rec_completas[-2] if len(capas_rec_completas) > 1 else dim_rec_in,
            1,
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
        rep_c = self._representacion_cliente(cliente, features_cliente)

        # Cabeza CLV
        clv_pred = self.cabeza_clv(rep_c).squeeze(-1)  # (B,)

        # Cabeza Recomendación
        e_p = self.emb_producto(producto)               # (B, dim_embedding)
        rec_in = torch.cat([rep_c, e_p], dim=-1)        # (B, dim_tronco_out + dim_emb)
        rec_hidden = self.cabeza_rec(rec_in)             # (B, capas_rec[-1]) o (B, dim_rec_in)
        rec_logit = self.cabeza_rec_out(rec_hidden).squeeze(-1)  # (B,)

        return clv_pred, rec_logit

    def _representacion_cliente(
        self,
        cliente: torch.Tensor,
        features_cliente: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calcula la representación del cliente (embedding + tronco).

        Parámetros
        ----------
        cliente : Tensor[long] shape (B,)
        features_cliente : Tensor[float] shape (B, n_features_cliente) | None

        Devuelve
        -------
        rep_c : Tensor[float] shape (B, dim_tronco_out)
        """
        e_c = self.emb_cliente(cliente)  # (B, dim_embedding)

        if self.usar_features:
            if features_cliente is None:
                features_cliente = torch.zeros(
                    e_c.size(0), self.n_features_cliente,
                    device=e_c.device, dtype=e_c.dtype,
                )
            x_c = torch.cat([e_c, features_cliente], dim=-1)
        else:
            x_c = e_c

        return self.tronco(x_c)

    # ------------------------------------------------------------------ #
    #  Métodos auxiliares                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_clv(
        self,
        cliente: torch.Tensor,
        features_cliente: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predice el CLV para un batch de clientes sin necesitar producto.

        A diferencia del ``forward`` principal, este método NO usa ningún
        producto de relleno: calcula directamente la representación del cliente
        y aplica la cabeza CLV. Es el método correcto para inferencia de CLV.

        Parámetros
        ----------
        cliente : Tensor[long] shape (B,)
            Índices de clientes.
        features_cliente : Tensor[float] shape (B, n_features_cliente) | None

        Devuelve
        -------
        clv_pred : Tensor[float] shape (B,)
            Predicción de CLV en escala log1p.
        """
        rep_c = self._representacion_cliente(cliente, features_cliente)
        return self.cabeza_clv(rep_c).squeeze(-1)

    def get_customer_embeddings(self) -> torch.Tensor:
        """Devuelve la matriz de embeddings de clientes (n_clientes, dim_embedding)."""
        return self.emb_cliente.weight.detach().cpu()

    def get_product_embeddings(self) -> torch.Tensor:
        """Devuelve la matriz de embeddings de productos (n_productos, dim_embedding)."""
        return self.emb_producto.weight.detach().cpu()

    @torch.no_grad()
    def score_todos_los_productos(
        self,
        customer_idx: int,
        device: torch.device | None = None,
        features_cliente: torch.Tensor | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Puntúa todos los productos para un cliente dado (con batching).

        Esta es la función centralizada que deben usar ``rec_metrics`` y
        ``export`` para evitar duplicación de lógica. Maneja automáticamente
        el batching para no provocar OOM en GPU.

        Parámetros
        ----------
        customer_idx : int
            Índice del cliente.
        device : torch.device | None
            Dispositivo de cómputo. Si None, usa el dispositivo del modelo.
        features_cliente : Tensor[float] shape (n_features_cliente,) | None
            Features del cliente (sin dimensión batch).
        batch_size : int
            Número de productos por batch.

        Devuelve
        -------
        logits : Tensor[float] shape (n_productos,)
            Logits de compra para cada producto.
        """
        if device is None:
            device = next(self.parameters()).device

        c_tensor = torch.tensor([customer_idx], dtype=torch.long, device=device)
        feat_tensor = None
        if self.usar_features and features_cliente is not None:
            feat_tensor = features_cliente.unsqueeze(0).to(device)

        # 1 sola pasada por el tronco del cliente
        rep_c = self._representacion_cliente(c_tensor, feat_tensor)  # (1, dim_tronco)
        n = self.n_productos
        emb_p = self.emb_producto.weight.to(device)                   # (n, dim_emb)

        # Puntuación vectorial completa en GPU
        x = torch.cat([rep_c.expand(n, -1), emb_p], dim=-1)
        rec_hidden = self.cabeza_rec(x)
        logits = self.cabeza_rec_out(rec_hidden).squeeze(-1)  # (n,)
        return logits.cpu()

    # Alias para compatibilidad hacia atrás
    score_all_products = score_todos_los_productos

    @classmethod
    def desde_config(cls, cfg: dict[str, Any]) -> "MultiTaskNet":
        """Construye la red a partir del dict de configuración."""
        m = cfg["modelo"]
        return cls(
            n_clientes=m["n_clientes"],
            n_productos=m["n_productos"],
            dim_embedding=m["dim_embedding"],
            capas_tronco=m.get("capas_tronco", [128, 64]),
            capas_rec=m.get("capas_rec", [64]),
            dropout=m["dropout"],
            usar_layer_norm=m["usar_layer_norm"],
            usar_features=m["usar_features"],
            n_features_cliente=m.get("n_features_cliente", 9),
        )
