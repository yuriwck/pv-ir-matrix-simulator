#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_efficientnet.py
========================

Avalia um modelo EfficientNet-B0 (PyTorch) treinado para classificação binária
(Normal vs Defect) em imagens térmicas pré-processadas (grayscale 64x64).

O script:
- Carrega um dataset no formato ImageFolder (raiz com subpastas por classe);
- Reconstrói a arquitetura do modelo (1 canal) e carrega pesos (.pth);
- Executa inferência no dataset completo (shuffle=False);
- Calcula métricas (accuracy, precision, recall, F1);
- Gera relatório de classificação e matrizes de confusão (absoluta e normalizada);
- Salva uma figura PNG com as matrizes lado a lado.

Uso (padrão):
    python evaluate_efficientnet.py

Uso (opcional, via CLI):
    python evaluate_efficientnet.py --dataset_dir dataset_simulado_v2 --model_path best_model_efficientnet_b0.pth

Notas importantes:
- Este arquivo mantém a lógica original de avaliação (softmax + threshold no índice 1).
- A normalização (mean/std) deve ser consistente com o pipeline de treinamento.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Configurações padrão (mantidas da versão original)
# =============================================================================
DEFAULT_DATASET_DIR = Path("dataset_simulado_v2")
DEFAULT_MODEL_PATH = Path("best_model_efficientnet_b0.pth")
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_CLASSES = 2
DEFAULT_THRESHOLD = 0.5
DEFAULT_OUTPUT_FIGURE = Path("matriz_confusao_combinada.png")


def get_device() -> str:
    """Retorna o dispositivo de execução (CUDA se disponível, caso contrário CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_transform(mean: float, std: float) -> transforms.Compose:
    """
    Constrói a transformação aplicada às imagens de entrada.

    Args:
        mean: Média usada na normalização (escala [0, 1]).
        std: Desvio padrão usado na normalização (escala [0, 1]).

    Returns:
        Pipeline de transforms do torchvision.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )


def build_model(num_classes: int) -> torch.nn.Module:
    """
    Constrói a EfficientNet-B0 adaptada para entrada em 1 canal e saída `num_classes`.

    Args:
        num_classes: Número de classes na camada final.

    Returns:
        Instância do modelo.
    """
    model = models.efficientnet_b0(weights=None)

    # Ajuste da primeira convolução para 1 canal (mantém hiperparâmetros originais).
    first_conv: nn.Conv2d = model.features[0][0]  # type: ignore[assignment]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )

    # Ajuste do classificador para o número de classes desejado.
    last_fc: nn.Linear = model.classifier[1]  # type: ignore[assignment]
    model.classifier[1] = nn.Linear(last_fc.in_features, num_classes)

    return model


def load_weights(model: torch.nn.Module, model_path: Path, device: str) -> None:
    """
    Carrega o state_dict do arquivo .pth para o modelo.

    Args:
        model: Modelo já construído com a arquitetura compatível.
        model_path: Caminho para o arquivo .pth.
        device: "cpu" ou "cuda".

    Raises:
        FileNotFoundError: Se `model_path` não existir.
        RuntimeError: Se o state_dict for incompatível com a arquitetura.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """
    Executa inferência em `loader` e retorna rótulos verdadeiros e preditos.

    Ponto crítico:
        - A decisão é feita por threshold em `probs[:, 1]` (classe índice 1),
          preservando o comportamento original.

    Args:
        model: Modelo em modo eval().
        loader: DataLoader do dataset.
        device: Dispositivo de execução.
        threshold: Limiar aplicado à probabilidade da classe 1.

    Returns:
        (y_true, y_pred) como listas de inteiros.
    """
    y_true: List[int] = []
    y_pred: List[int] = []

    pbar = tqdm(loader, desc="Inferência", unit="batch", leave=True)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] > threshold).long()

            y_true_batch = labels.detach().cpu().numpy().tolist()
            y_pred_batch = preds.detach().cpu().numpy().tolist()

            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)

            # Métrica parcial (apenas para acompanhar progresso; não afeta a avaliação final).
            partial_acc = (np.array(y_true) == np.array(y_pred)).mean()
            pbar.set_postfix(acc=f"{partial_acc:.3f}")

    return y_true, y_pred


def compute_and_log_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> None:
    """
    Calcula métricas e imprime relatório detalhado.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        class_names: Nomes das classes no dataset.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    LOGGER.info("=== MÉTRICAS ===")
    LOGGER.info("Acurácia : %.4f", acc)
    LOGGER.info("Precisão : %.4f", prec)
    LOGGER.info("Recall   : %.4f", rec)
    LOGGER.info("F1-score : %.4f", f1)

    LOGGER.info("=== RELATÓRIO DETALHADO ===\n%s", classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrices(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """
    Plota e salva as matrizes de confusão absoluta e normalizada (recall por classe).

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        class_names: Nomes das classes.
        output_path: Caminho do arquivo PNG de saída.
    """
    cm_abs = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.5))

    sns.heatmap(
        cm_abs,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Contagem Absoluta", fontsize=12)
    axes[0].set_ylabel("Classe Real", fontsize=11)
    axes[0].set_xlabel("Classe Predita", fontsize=11)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Normalizada (Recall por Classe)", fontsize=12)
    axes[1].set_xlabel("Classe Predita", fontsize=11)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info("Imagem salva como: %s", output_path)
    plt.show()


def parse_args() -> argparse.Namespace:
    """Define e processa argumentos de linha de comando (mantendo defaults originais)."""
    parser = argparse.ArgumentParser(
        description="Avalia um modelo EfficientNet-B0 em dataset ImageFolder e gera métricas + matriz de confusão.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR, help="Diretório raiz do dataset (ImageFolder).")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Caminho do checkpoint .pth.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size para inferência.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold aplicado à probabilidade da classe 1.")
    parser.add_argument("--mean", type=float, default=0.6193, help="Média usada na normalização.")
    parser.add_argument("--std", type=float, default=0.1548, help="Desvio padrão usado na normalização.")
    parser.add_argument("--output_figure", type=Path, default=DEFAULT_OUTPUT_FIGURE, help="Arquivo de saída (PNG) da matriz de confusão.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()
    device = get_device()

    transform = build_transform(mean=args.mean, std=args.std)

    dataset = datasets.ImageFolder(str(args.dataset_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    class_names = dataset.classes

    LOGGER.info("Classes: %s", class_names)
    LOGGER.info("Device: %s", device)

    model = build_model(num_classes=DEFAULT_NUM_CLASSES)
    load_weights(model, args.model_path, device=device)

    model.to(device)
    model.eval()

    y_true, y_pred = run_inference(model=model, loader=loader, device=device, threshold=args.threshold)
    compute_and_log_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names)
    plot_confusion_matrices(y_true=y_true, y_pred=y_pred, class_names=class_names, output_path=args.output_figure)


if __name__ == "__main__":
    main()
