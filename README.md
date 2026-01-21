# pv-ir-matrix-simulator
Simulador de termografia para módulos fotovoltaicos: gera matrizes térmicas sintéticas no formato sensores × deslocamento para avaliação de detecção de anomalias.
# Simulador IR + Geração de Dataset + Avaliação (TCC)

Este repositório reúne scripts Python utilizados como entregável do meu TCC para:
1) simular matrizes térmicas no formato **sensores × deslocamento** (módulos fotovoltaicos);
2) gerar um dataset de classificação **Normal vs Defect** em estrutura compatível com `ImageFolder`;
3) avaliar um modelo **EfficientNet-B0 (PyTorch)** no dataset gerado (métricas + matriz de confusão).

> Observação: este repositório foi organizado para leitura por avaliadores. Os scripts são focados em rastreabilidade e reprodutibilidade dentro do pipeline experimental.

---

## Conteúdo do repositório

- `matrix_generator_v3_test.py`  
  Simulador IR: gera fundo saudável, injeta defeitos sintéticos e converte a matriz térmica para imagem 8-bit (ex.: estilo *centered* / AGC). Inclui um teste visual (plot) quando executado diretamente.

- `build_classification_dataset_test.py`  
  Gera um dataset balanceado **normal/defect**, salvando imagens PNG e metadados (`metadata.csv`, `metadata.json`, `summary.txt`) em estrutura compatível com `torchvision.datasets.ImageFolder`.

- `evaluate_efficientnet.py`  
  Avalia um modelo EfficientNet-B0 treinado para **Normal vs Defect**. Calcula métricas e salva uma figura PNG com matrizes de confusão (absoluta e normalizada).

---

## Requisitos

- Python **3.10+** é recomendado para compatibilidade com versões estáveis recentes do PyTorch (ver nota no site do PyTorch).  
  Se o seu ambiente usar outra versão de Python, instale uma versão compatível do PyTorch (ver “Previous Versions”).

- Dependências Python estão listadas em `requirements.txt`.

---

## Instalação

### 1) Criar ambiente virtual (recomendado)

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
