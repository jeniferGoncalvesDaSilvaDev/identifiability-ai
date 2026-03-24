"""
RepresentationMetrics
=====================
Análise de identificabilidade e estrutura informacional
para qualquer modelo PyTorch via hooks automáticos.

Métricas:
    - Rank efetivo (SVD)
    - Decaimento espectral
    - Taxa de compressão
    - Nulidade funcional
    - Sensibilidade (norma do gradiente)
    - Impacto da compressão na saída

Uso rápido:
    analyzer = ModelAnalyzer(model)
    report = analyzer.analyze(x, output_layer="layer4")
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Estrutura de resultado
# ─────────────────────────────────────────────

@dataclass
class LayerReport:
    layer_name: str
    effective_rank: float
    spectral_decay: np.ndarray          # valores singulares normalizados
    compression_ratio: float            # k / d para threshold dado
    dims_needed: int                    # k dimensões necessárias
    functional_nullity: float           # fração de direções "invisíveis"
    sensitivity: Optional[float]        # norma do jacobiano (None se não aplicável)
    compression_delta: Optional[float]  # impacto na saída ao comprimir
    singular_values: np.ndarray         # espectro bruto

    def summary(self) -> str:
        lines = [
            f"\n── {self.layer_name} ──",
            f"  Rank efetivo        : {self.effective_rank:.2f}",
            f"  Dimensões necessárias: {self.dims_needed} / {len(self.singular_values)}",
            f"  Taxa de compressão  : {self.compression_ratio:.1%}",
            f"  Nulidade funcional  : {self.functional_nullity:.1%}",
        ]
        if self.sensitivity is not None:
            lines.append(f"  Sensibilidade       : {self.sensitivity:.4f}")
        if self.compression_delta is not None:
            lines.append(f"  Δ saída (compressão): {self.compression_delta:.4f}")
        return "\n".join(lines)


@dataclass
class AnalysisReport:
    model_name: str
    layers: Dict[str, LayerReport] = field(default_factory=dict)

    def print(self):
        print(f"\n{'='*50}")
        print(f"  Modelo: {self.model_name}")
        print(f"{'='*50}")
        for report in self.layers.values():
            print(report.summary())
        print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# Métricas base (stateless, puro tensor)
# ─────────────────────────────────────────────

class RepresentationMetrics:
    """
    Cálculo puro de métricas sobre embeddings.
    Independente de modelo — recebe tensores direto.
    """

    @staticmethod
    def compute_svd(X: torch.Tensor) -> torch.Tensor:
        """Retorna valores singulares de X (centrado)."""
        X = X.float()
        X = X - X.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        return S

    @staticmethod
    def effective_rank(S: torch.Tensor) -> float:
        """
        Rank efetivo: (Σ sᵢ)² / Σ sᵢ²
        Mede dimensionalidade real da representação.
        """
        num = torch.sum(S) ** 2
        den = torch.sum(S ** 2) + 1e-8
        return (num / den).item()

    @staticmethod
    def spectral_decay(S: torch.Tensor) -> np.ndarray:
        """Valores singulares normalizados."""
        s = S / (S.sum() + 1e-8)
        return s.cpu().numpy()

    @staticmethod
    def compression_ratio(
        S: torch.Tensor,
        variance_threshold: float = 0.90,
    ) -> tuple[float, int]:
        """
        Menor k para preservar `variance_threshold` da variância.
        Retorna: (k/d, k)
        """
        var = S ** 2
        total = var.sum()
        cumulative = torch.cumsum(var, dim=0)
        k = int(torch.searchsorted(cumulative, variance_threshold * total).item()) + 1
        k = min(k, len(S))
        return k / len(S), k

    @staticmethod
    def functional_nullity(
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        h: torch.Tensor,
        epsilon: float = 1e-3,
        n_directions: int = 32,
    ) -> float:
        """
        Fração de direções aleatórias que não afetam a saída.
        forward_fn: função que mapeia embedding → saída final.
        """
        h = h.detach()
        base = forward_fn(h)

        null_count = 0
        for _ in range(n_directions):
            v = F.normalize(torch.randn_like(h), dim=-1)
            delta = torch.norm(
                forward_fn(h + epsilon * v) - base, dim=-1
            ).mean().item()
            if delta < 1e-4:
                null_count += 1

        return null_count / n_directions

    @staticmethod
    def sensitivity(
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        h: torch.Tensor,
    ) -> float:
        """
        Norma média do gradiente da saída em relação ao embedding.
        Mede o quanto pequenas variações no embedding alteram a saída.
        """
        h_grad = h.clone().detach().float().requires_grad_(True)
        out = forward_fn(h_grad)

        # proxy escalar: máximo logit médio
        scalar = out.max(dim=-1)[0].mean()
        scalar.backward()

        if h_grad.grad is None:
            return 0.0

        return torch.norm(h_grad.grad, dim=-1).mean().item()

    @staticmethod
    def compression_delta(
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        h: torch.Tensor,
        k: int,
    ) -> float:
        """
        Impacto na saída ao projetar o embedding nas k componentes principais.
        Mede quanto de informação semântica está nas dimensões redundantes.
        """
        h = h.float()
        h_centered = h - h.mean(dim=0, keepdim=True)

        _, _, Vt = torch.linalg.svd(h_centered, full_matrices=False)
        V_k = Vt[:k].T
        h_proj = h_centered @ V_k @ V_k.T + h.mean(dim=0, keepdim=True)

        with torch.no_grad():
            out_orig = forward_fn(h)
            out_proj = forward_fn(h_proj)

        return torch.norm(out_orig - out_proj, dim=-1).mean().item()


# ─────────────────────────────────────────────
# Hook manager — captura ativações de qualquer modelo
# ─────────────────────────────────────────────

class HookManager:
    """Registra forward hooks em camadas nomeadas."""

    def __init__(self):
        self._hooks: list = []
        self.activations: Dict[str, torch.Tensor] = {}

    def register(self, module: torch.nn.Module, name: str):
        def hook(_, __, output):
            # Achata tudo exceto batch dim → (batch, d)
            self.activations[name] = _flatten(output.detach())

        h = module.register_forward_hook(hook)
        self._hooks.append(h)

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @contextmanager
    def capture(self):
        try:
            yield self
        finally:
            self.remove_all()


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) → (B, C*H*W) etc."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    return t.flatten(start_dim=1)


# ─────────────────────────────────────────────
# Analisador principal
# ─────────────────────────────────────────────

class ModelAnalyzer:
    """
    Analisa qualquer modelo PyTorch.

    Parâmetros:
        model        : nn.Module qualquer
        output_head  : função que mapeia embedding → logits/probs.
                       Se None, tenta usar model.fc / model.head / model.classifier.
        device       : torch.device
        variance_thr : threshold para compression_ratio (padrão 90%)

    Exemplo:
        analyzer = ModelAnalyzer(resnet)
        report = analyzer.analyze(x, layers=["layer3", "layer4"])
        report.print()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        output_head: Optional[Callable] = None,
        device: str | torch.device = "cpu",
        variance_thr: float = 0.90,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.variance_thr = variance_thr
        self.metrics = RepresentationMetrics()

        self._output_head = output_head or self._infer_head()

    # ── forward head ──

    def _infer_head(self) -> Optional[Callable]:
        """Tenta descobrir automaticamente a cabeça de classificação."""
        for attr in ("fc", "head", "classifier", "output"):
            head = getattr(self.model, attr, None)
            if head is not None and callable(head):
                warnings.warn(
                    f"[ModelAnalyzer] Usando '{attr}' como output_head automático.",
                    stacklevel=2,
                )
                return head
        warnings.warn(
            "[ModelAnalyzer] Nenhum output_head encontrado. "
            "Métricas funcionais serão puladas.",
            stacklevel=2,
        )
        return None

    def _safe_forward(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self._output_head is None:
            return None
        try:
            with torch.no_grad():
                return self._output_head(h.float())
        except Exception:
            return None

    # ── listagem de camadas ──

    def list_layers(self) -> Dict[str, torch.nn.Module]:
        """Retorna dicionário {nome: módulo} de todas as camadas."""
        return dict(self.model.named_modules())

    def print_layers(self):
        for name, mod in self.list_layers().items():
            print(f"  {name or '<root>':<40} {type(mod).__name__}")

    # ── análise principal ──

    def analyze(
        self,
        x: torch.Tensor,
        layers: Optional[list[str]] = None,
        n_directions: int = 32,
        epsilon: float = 1e-3,
    ) -> AnalysisReport:
        """
        Roda análise completa.

        Parâmetros:
            x          : input batch (B, ...)
            layers     : lista de nomes de camadas para analisar.
                         Se None, usa todas as camadas folha com parâmetros.
            n_directions: número de direções para nulidade funcional
            epsilon    : perturbação para nulidade funcional

        Retorna:
            AnalysisReport com LayerReport por camada
        """
        x = x.to(self.device)
        self.model.eval()

        named = dict(self.model.named_modules())

        if layers is None:
            layers = self._default_layers(named)

        hook_mgr = HookManager()
        for name in layers:
            if name not in named:
                warnings.warn(f"[ModelAnalyzer] Camada '{name}' não encontrada.")
                continue
            hook_mgr.register(named[name], name)

        with torch.no_grad():
            self.model(x)

        hook_mgr.remove_all()

        report = AnalysisReport(model_name=type(self.model).__name__)

        for name in layers:
            if name not in hook_mgr.activations:
                continue
            h = hook_mgr.activations[name].to(self.device)
            report.layers[name] = self._analyze_layer(
                name, h, n_directions, epsilon
            )

        return report

    def _analyze_layer(
        self,
        name: str,
        h: torch.Tensor,
        n_directions: int,
        epsilon: float,
    ) -> LayerReport:
        S = self.metrics.compute_svd(h)
        eff_rank = self.metrics.effective_rank(S)
        decay = self.metrics.spectral_decay(S)
        comp_ratio, k = self.metrics.compression_ratio(S, self.variance_thr)

        # métricas funcionais (opcionais)
        if self._output_head is not None:
            forward_fn = lambda emb: F.softmax(self._output_head(emb.float()), dim=-1)
            try:
                nullity = self.metrics.functional_nullity(
                    forward_fn, h, epsilon, n_directions
                )
            except Exception:
                nullity = float("nan")

            try:
                sens = self.metrics.sensitivity(forward_fn, h)
            except Exception:
                sens = None

            try:
                delta = self.metrics.compression_delta(forward_fn, h, k)
            except Exception:
                delta = None
        else:
            nullity = float("nan")
            sens = None
            delta = None

        return LayerReport(
            layer_name=name,
            effective_rank=eff_rank,
            spectral_decay=decay,
            compression_ratio=comp_ratio,
            dims_needed=k,
            functional_nullity=nullity,
            sensitivity=sens,
            compression_delta=delta,
            singular_values=S.cpu().numpy(),
        )

    @staticmethod
    def _default_layers(named: Dict) -> list[str]:
        """Seleciona camadas folha com parâmetros."""
        result = []
        for name, mod in named.items():
            if name == "":
                continue
            has_params = any(True for _ in mod.parameters(recurse=False))
            is_leaf = len(list(mod.children())) == 0
            if is_leaf and has_params:
                result.append(name)
        return result


# ─────────────────────────────────────────────
# Utilitários de visualização (opcional)
# ─────────────────────────────────────────────

def plot_spectral_decay(report: AnalysisReport, max_layers: int = 4):
    """
    Plota decaimento espectral de cada camada analisada.
    Requer matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib não instalado.")
        return

    layers = list(report.layers.values())[:max_layers]
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4))
    if len(layers) == 1:
        axes = [axes]

    for ax, lr in zip(axes, layers):
        ax.bar(range(len(lr.spectral_decay)), lr.spectral_decay, color="#4f9eff")
        ax.set_title(f"{lr.layer_name}\nrank_eff={lr.effective_rank:.1f}")
        ax.set_xlabel("componente")
        ax.set_ylabel("variância relativa")
        ax.axvline(lr.dims_needed - 1, color="red", linestyle="--", label=f"k={lr.dims_needed}")
        ax.legend()

    plt.suptitle(f"Decaimento Espectral — {report.model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig("spectral_decay.png", dpi=150)
    plt.show()
    print("Salvo: spectral_decay.png")


# ─────────────────────────────────────────────
# Demo rápida
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn as nn

    # modelo mínimo de exemplo
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
            )
            self.classifier = nn.Linear(64, 10)

        def forward(self, x):
            return self.classifier(self.features(x))

    model = SimpleNet()
    x = torch.randn(32, 64)

    analyzer = ModelAnalyzer(model, device="cpu")

    print("\nCamadas disponíveis:")
    analyzer.print_layers()

    report = analyzer.analyze(x, layers=["features.0", "features.2"])
    report.print()

    plot_spectral_decay(report)
