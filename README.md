# 🧶 Crochet Classifier

Um classificador de imagens (visão computacional) para detectar erros em
peças de crochê — ponto pulado, tensão irregular, borda deformada, entre
outros — construído com PyTorch.

> ⚠️ **Estágio atual: dados sintéticos (mock).** As imagens usadas até
> agora são geradas artificialmente por script, não são fotos reais de
> crochê. Veja a seção [Por que dados mockados?](#-por-que-dados-mockados)
> para entender o motivo.

---

## 📌 Sobre o projeto

O objetivo final é treinar um modelo capaz de olhar para a foto de uma
amostra de crochê e classificá-la como **correta** ou **com erro**
(ponto pulado, aumento indevido, diminuição indevida, tensão irregular,
borda deformada, etc.).

Este é um projeto de longo prazo, dividido em fases:

- **Fase 1 (atual)** — Prova de conceito: um único fio, agulha e ponto,
  ~100 amostras físicas, classificação binária (correto/erro)
- **Fase 2** — Pipeline de código completo (carregamento, augmentation,
  treino, avaliação)
- **Fase 3** — Expansão do dataset (mais fios, cores, pontos, agulhas)
- **Fases futuras** — Localização do erro na imagem, classificação do
  tipo específico de erro, reconhecimento estrutural (carreiras,
  contagem de pontos), e eventualmente um sistema mais completo de apoio
  à modelagem em crochê

## 🖼️ Por que dados mockados?

Fabricar e fotografar 100 peças físicas de crochê, com protocolo
padronizado (fundo, iluminação, ângulo), leva tempo — e não precisa
travar o desenvolvimento do código.

Existe uma diferença entre **"o código funciona sem bugs"** e
**"o modelo aprende algo real sobre crochê"**. A primeira parte pode
(e deve) ser validada agora, em paralelo à fabricação das peças reais.

Por isso, o pipeline atual usa imagens **sintéticas**, geradas por
[`scripts/gerar_mock.py`](scripts/gerar_mock.py): padrões simples de
linhas repetidas simulam a classe "correto", e o mesmo padrão com uma
linha deslocada ou faltando simula a classe "erro". Essas imagens não
ensinam nada sobre crochê real — elas só replicam a mesma estrutura de
pastas e formato de arquivo que as fotos reais vão ter.

**Quando as fotos reais chegarem:** basta substituir o conteúdo das
pastas `dataset/correto/` e `dataset/erro/` pelas fotos de verdade e
rodar o mesmo pipeline, sem alterar nenhum código.

## 📁 Estrutura do projeto
crochet-classifier/
├── dataset/
│   ├── correto/              # imagens (mock, por enquanto) da classe "correto"
│   ├── erro/                 # imagens (mock, por enquanto) da classe "erro"
│   └── labels_mock.csv       # planilha de rotulagem
├── dataset_teste_externo/    # lote de holdout, nunca usado em treino/validação
├── notebooks/
│   └── 01_carregamento.ipynb # pipeline de carregamento de imagens
├── scripts/
│   └── gerar_mock.py         # gerador de dados sintéticos
├── .gitignore
└── README.md
## 🛠️ Stack técnica

- **Python** 3.12
- **PyTorch** + **torchvision** (modelo e pipeline de dados)
- **OpenCV** (geração das imagens sintéticas)
- **Matplotlib** (visualização)
- **Jupyter Notebook** (desenvolvimento interativo)
- Treinamento em **CPU** (dataset pequeno, ~100-200 imagens)

## 🚀 Como rodar localmente

```powershell
# Clonar o repositório
git clone https://github.com/isafranzon/crochet-classifier.git
cd crochet-classifier

# Criar e ativar o ambiente virtual (Python 3.12)
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar as dependências
python -m pip install torch torchvision opencv-python matplotlib notebook ipykernel

# Gerar o dataset sintético (mock)
python scripts/gerar_mock.py

# Abrir os notebooks em notebooks/ (recomendado: VS Code com a extensão Jupyter)
```

## 📊 Progresso atual

- [x] Ambiente configurado (venv, PyTorch, OpenCV)
- [x] Gerador de dados sintéticos (`gerar_mock.py`)
- [x] Pipeline de carregamento de imagens (`ImageFolder` + `DataLoader`)
- [ ] Data augmentation
- [ ] Divisão treino / validação / teste
- [ ] Loop de treinamento
- [ ] Avaliação (matriz de confusão, métricas)
- [ ] Substituição dos dados mockados pelas fotos reais
- [ ] Treinamento com dados reais

## 📷 Exemplo de saída do pipeline

*(imagem de exemplo aqui — 4 amostras "correto" e 4 "erro" carregadas e visualizadas via matplotlib)*

---

Projeto em desenvolvimento contínuo. Sugestões e feedback são bem-vindos!