#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_classification_dataset_test.py
====================================

Gera um dataset balanceado (Normal vs Defect) a partir do simulador definido em
``matrix_generator_v3_test.py``.

Estrutura de saída (ImageFolder-friendly):
    out_dir/
        normal/
            normal_00000.png
            ...
        defect/
            defect_00000.png
            ...
        metadata.csv
        metadata.json
        summary.txt

Decisões de projeto (mantidas da versão original)
-------------------------------------------------
- NORMAL:
    Usa ``healthy_background()`` para produzir um fundo "plano", reduzindo chance de
    falso-positivo induzido por gradientes artificiais.

- DEFECT:
    Parte de um fundo saudável, injeta de 1 a ``max_defs`` defeitos via ``DEFECT_GENERATORS``.

- Normalização "centered":
    A baseline é calculada no fundo saudável (mediana de mat0), e não na matriz
    já defeituosa (mat). Essa decisão mantém o contraste relativo do defeito.

Exemplo:
    python build_classification_dataset_test.py \
        --out_dir dataset_simulado_v2 \
        --n_normals 1000 \
        --n_defects 1000 \
        --max_defs 1 \
        --seed 42 \
        --delta_white 15.0 \
        --delta_black -15.0

Notas importantes:
- Este script não altera lógica do simulador; apenas orquestra geração e escrita.
- Para reprodutibilidade: use ``--seed`` diferente de 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

from matrix_generator_v3_test import DEFECT_GENERATORS, gerar_realista, healthy_background

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Defeitos selecionados (após filtragem)
# =============================================================================
SELECTED_DEFECT_TYPES: List[str] = [
    "cell",
    "hot-spot-multi",
    "cracking",
    "shadowing",
    "soiling",
    "diode",
    "diode-multi",
]

# =============================================================================
# Validação: defeitos disponíveis (executada na importação para falha precoce)
# =============================================================================
AVAILABLE_DEFECTS = [k for k, v in DEFECT_GENERATORS.items() if v is not None]
for defect in SELECTED_DEFECT_TYPES:
    if defect not in AVAILABLE_DEFECTS:
        raise RuntimeError(
            f"Defeito '{defect}' não está disponível. Disponíveis: {AVAILABLE_DEFECTS}"
        )


def generate_classification_dataset(
    out_dir: str,
    n_normals: int,
    n_defects: int,
    defect_types: List[str],
    base_temp_range: Tuple[float, float] = (25.0, 40.0),
    max_defs: int = 3,
    seed: int = 0,
    delta_white: float = 5.0,
    delta_black: float = -15.0,
) -> None:
    """
    Gera dataset (PNG + metadados) para classificação binária normal/defect.

    Args:
        out_dir: Pasta de saída.
        n_normals: Número de amostras normais.
        n_defects: Número de amostras defeituosas.
        defect_types: Lista de defeitos (chaves de DEFECT_GENERATORS) elegíveis para injeção.
        base_temp_range: Intervalo da temperatura base (°C).
        max_defs: Máximo de defeitos por amostra defeituosa.
        seed: Semente. Por compatibilidade: 0 => aleatória (não fixa).
        delta_white: Parâmetro de normalização centered (°C).
        delta_black: Parâmetro de normalização centered (°C, negativo).

    Returns:
        None. Escreve arquivos em disco.

    Raises:
        RuntimeError: Se uma amostra "defect" for gerada sem defeitos (condição inesperada).
    """
    # Ponto crítico (reprodutibilidade):
    # Mantém a regra original: seed==0 => não fixa seeds; seed!=0 fixa numpy e random.
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    normal_dir = os.path.join(out_dir, "normal")
    defect_dir = os.path.join(out_dir, "defect")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)

    metadata: List[Dict[str, object]] = []

    LOGGER.info("Gerando %d amostras normais...", n_normals)
    for i in range(n_normals):
        base_temp = float(np.random.uniform(*base_temp_range))

        # Normal: fundo saudável "plano"
        mat0 = healthy_background(base_temp)

        # Baseline: mediana do fundo saudável
        baseline = float(np.median(mat0))

        img = gerar_realista(
            mat0,
            color=False,
            style="centered",
            delta_white=delta_white,
            delta_black=delta_black,
            baseline=baseline,
        )

        filename = f"normal_{i:05d}.png"
        filepath = os.path.join(normal_dir, filename)
        cv2.imwrite(filepath, img)

        metadata.append(
            {
                "id": filename,
                "class": "normal",
                "defect_list": [],
                "delta_peaks": [],
                "params": [],
                "base_temp": base_temp,
            }
        )

        if (i + 1) % 100 == 0:
            LOGGER.info("  Normais: %d/%d", i + 1, n_normals)

    LOGGER.info("Gerando %d amostras defeituosas...", n_defects)
    for i in range(n_defects):
        base_temp = float(np.random.uniform(*base_temp_range))

        # Base saudável primeiro; defeitos são injetados em uma cópia.
        mat0 = healthy_background(base_temp)
        mat = mat0.copy()

        # Sorteia de 1 a max_defs defeitos.
        num_defs = random.randint(1, max_defs)
        chosen = random.choices(defect_types, k=num_defs)

        specs = []
        for dtype in chosen:
            func = DEFECT_GENERATORS.get(dtype)
            if func is not None:
                specs.append(func(mat))

        # Segurança (mantida): nunca salvar defect sem defeito.
        if len(specs) == 0:
            raise RuntimeError(
                "Gerou amostra defect sem defeitos. Verifique DEFECT_GENERATORS/seleção."
            )

        # Baseline do fundo saudável (mat0), não do mat com defeito.
        baseline = float(np.median(mat0))

        img = gerar_realista(
            mat,
            color=False,
            style="centered",
            delta_white=delta_white,
            delta_black=delta_black,
            baseline=baseline,
        )

        filename = f"defect_{i:05d}.png"
        filepath = os.path.join(defect_dir, filename)
        cv2.imwrite(filepath, img)

        metadata.append(
            {
                "id": filename,
                "class": "defect",
                "defect_list": [s.dtype for s in specs],
                "delta_peaks": [float(s.delta_t_peak) for s in specs],
                "params": [s.params for s in specs],
                "base_temp": base_temp,
            }
        )

        if (i + 1) % 100 == 0:
            LOGGER.info("  Defeitos: %d/%d", i + 1, n_defects)

    _write_metadata(out_dir=out_dir, metadata=metadata)
    _write_summary(
        out_dir=out_dir,
        n_normals=n_normals,
        n_defects=n_defects,
        defect_types=defect_types,
        base_temp_range=base_temp_range,
        max_defs=max_defs,
        seed=seed,
        delta_white=delta_white,
        delta_black=delta_black,
    )

    LOGGER.info("Dataset gerado com sucesso em '%s'", out_dir)
    LOGGER.info("  - Normais: %d amostras", n_normals)
    LOGGER.info("  - Defeitos: %d amostras", n_defects)
    LOGGER.info("  - Total: %d amostras", len(metadata))
    LOGGER.info("Arquivos:")
    LOGGER.info("  - %s/", normal_dir)
    LOGGER.info("  - %s/", defect_dir)
    LOGGER.info("  - %s", os.path.join(out_dir, "metadata.csv"))
    LOGGER.info("  - %s", os.path.join(out_dir, "metadata.json"))
    LOGGER.info("  - %s", os.path.join(out_dir, "summary.txt"))


def _write_metadata(out_dir: str, metadata: List[Dict[str, object]]) -> None:
    """
    Escreve metadata.csv e metadata.json.

    Args:
        out_dir: Diretório de saída.
        metadata: Lista de dicionários de metadados.
    """
    csv_path = os.path.join(out_dir, "metadata.csv")
    json_path = os.path.join(out_dir, "metadata.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv, fieldnames=["id", "class", "defect_list", "delta_peaks", "params", "base_temp"]
        )
        writer.writeheader()

        for m in metadata:
            row = dict(m)
            row["defect_list"] = ";".join(row["defect_list"])  # type: ignore[arg-type]
            row["delta_peaks"] = ";".join([f"{v:.2f}" for v in row["delta_peaks"]])  # type: ignore[arg-type]
            row["params"] = json.dumps(row["params"], ensure_ascii=False)  # type: ignore[arg-type]
            writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(metadata, f_json, ensure_ascii=False, indent=2)


def _write_summary(
    out_dir: str,
    n_normals: int,
    n_defects: int,
    defect_types: List[str],
    base_temp_range: Tuple[float, float],
    max_defs: int,
    seed: int,
    delta_white: float,
    delta_black: float,
) -> None:
    """
    Escreve summary.txt com parâmetros e composição do dataset.

    Args:
        out_dir: Diretório de saída.
        n_normals: Número de normais.
        n_defects: Número de defeituosas.
        defect_types: Tipos de defeito utilizados.
        base_temp_range: Intervalo da temperatura base.
        max_defs: Máximo de defeitos por amostra defect.
        seed: Semente (0 => aleatória).
        delta_white: Parâmetro de centered.
        delta_black: Parâmetro de centered.
    """
    summary_path = os.path.join(out_dir, "summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f_summary:
        f_summary.write("=" * 60 + "\n")
        f_summary.write("DATASET DE CLASSIFICAÇÃO - RESUMO\n")
        f_summary.write("=" * 60 + "\n\n")
        f_summary.write(f"Total de amostras: {n_normals + n_defects}\n")
        f_summary.write(f"  - Normais: {n_normals}\n")
        f_summary.write(f"  - Defeitos: {n_defects}\n\n")
        f_summary.write(f"Defeitos utilizados ({len(defect_types)} tipos):\n")
        for d in defect_types:
            f_summary.write(f"  - {d}\n")
        f_summary.write("\nParâmetros:\n")
        f_summary.write(f"  - Semente: {seed if seed else 'aleatória'}\n")
        f_summary.write(f"  - Máx. defeitos por amostra defeituosa: {max_defs}\n")
        f_summary.write(f"  - Faixa base_temp: {base_temp_range[0]}..{base_temp_range[1]} °C\n")
        f_summary.write(f"  - delta_white: {delta_white} °C\n")
        f_summary.write(f"  - delta_black: {delta_black} °C\n")


def parse_args() -> argparse.Namespace:
    """Configura e processa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera dataset classificado normal/defect a partir de matrix_generator_v3_test.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="dataset_simulado_v2",
        help="Pasta de saída (ImageFolder).",
    )
    parser.add_argument(
        "--n_normals",
        type=int,
        default=1000,
        help="Número de imagens normais.",
    )
    parser.add_argument(
        "--n_defects",
        type=int,
        default=1000,
        help="Número de imagens com defeito.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semente (0 = aleatória).",
    )
    parser.add_argument(
        "--max_defs",
        type=int,
        default=1,
        help="Máximo de defeitos por imagem defeituosa.",
    )
    parser.add_argument(
        "--delta_white",
        type=float,
        default=5.0,
        help="ΔT para saturação em branco (acima da baseline), em °C.",
    )
    parser.add_argument(
        "--delta_black",
        type=float,
        default=-15.0,
        help="ΔT para saturação em preto (abaixo da baseline), em °C.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    LOGGER.info("=" * 70)
    LOGGER.info("GERADOR DE DATASET PARA CLASSIFICAÇÃO (NORMAL vs DEFECT)")
    LOGGER.info("=" * 70)
    LOGGER.info("Configuração:")
    LOGGER.info("  - out_dir: %s", args.out_dir)
    LOGGER.info("  - n_normals: %d", args.n_normals)
    LOGGER.info("  - n_defects: %d", args.n_defects)
    LOGGER.info("  - max_defs: %d", args.max_defs)
    LOGGER.info("  - seed: %s", args.seed if args.seed else "aleatória")
    LOGGER.info("  - delta_white: %.3f", args.delta_white)
    LOGGER.info("  - delta_black: %.3f", args.delta_black)
    LOGGER.info("Defeitos incluídos:")
    for i, d in enumerate(SELECTED_DEFECT_TYPES, 1):
        LOGGER.info("  %2d. %s", i, d)

    generate_classification_dataset(
        out_dir=args.out_dir,
        n_normals=args.n_normals,
        n_defects=args.n_defects,
        defect_types=SELECTED_DEFECT_TYPES,
        max_defs=args.max_defs,
        seed=args.seed,
        delta_white=args.delta_white,
        delta_black=args.delta_black,
    )


if __name__ == "__main__":
    main()
