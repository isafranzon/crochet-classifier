"""
Gerador de imagens sinteticas (mock) para testar o pipeline
de classificacao correto/erro sem depender das fotos reais.

NAO ensina nada sobre crochê de verdade — so replica a mesma
estrutura de pastas e formato de arquivo que as fotos reais vao ter
(dataset/correto/, dataset/erro/), para que o pipeline de codigo
(carregamento, augmentation, split, treino, avaliacao) possa ser
testado agora, em paralelo a fabricacao/fotografia das pecas reais.

Quando as fotos reais chegarem: so trocar o conteudo das pastas
dataset/correto e dataset/erro pelas fotos reais e rodar o mesmo
codigo de pipeline, sem alterar nada nos notebooks.
"""

import os
import csv
import random
import numpy as np
import cv2

# ---------- CONFIGURACOES ----------

# Resolucao "crua" da imagem gerada. Escolhemos um valor MAIOR do que
# o tamanho final esperado pelo modelo (224x224, padrao ResNet/transfer
# learning) de proposito: assim o redimensionamento de verdade acontece
# dentro do pipeline (notebook 01_carregamento), nao aqui no gerador.
# Isso garante que o codigo de resize seja realmente testado, do mesmo
# jeito que vai precisar ser quando as fotos reais (que virao em alta
# resolucao, direto da camera/celular) chegarem.
RESOLUCAO = 512

# Volume alinhado ao dataset REAL da Fase 1 (ver PROJETO_CROCHE_IA.md,
# secao 3): 100 amostras totais, 50 corretas + 50 com erro.
# Decisao: usar o MESMO volume no mock, em vez de um numero menor (ex:
# 20 por classe, usado numa primeira versao deste script). Motivo:
# quando formos testar o DataLoader, o split treino/val/teste (item 3
# do checklist) e o tempo por epoca do treino (item 4), queremos que
# esses testes ja reflitam o volume real que o projeto vai ter --
# assim nao ha surpresa de comportamento quando o mock for substituido
# pelas fotos de verdade.
QTD_TREINO_POR_CLASSE = 50   # 50 correto + 50 erro = 100 no total

# Holdout = lote "de fora", nunca usado em treino/validacao/teste do
# pipeline principal. Serve para o teste 5 do PROJETO_CROCHE_IA.md
# (secao 8): "fotografar 5-10 pecas extras fora do dataset de treino
# para validar o modelo em dados nunca vistos". Escolhemos 13 por
# classe (26 no total) para ficar dentro dessa faixa de 5-10% a mais,
# proporcional ao novo volume de 100.
QTD_HOLDOUT_POR_CLASSE = 13  # 13 correto + 13 erro = 26 extras

# Seed fixa: garante reprodutibilidade. Rodando o script varias vezes,
# sempre gera exatamente as mesmas imagens (mesmas cores, espacamentos,
# posicoes de defeito). Importante para depuracao -- se algo quebrar
# no pipeline, sabemos que nao foi por causa de uma imagem "diferente"
# gerada aleatoriamente numa nova execucao.
SEED = 42

# Tipos de defeito simulados. Nao tem relacao real com defeitos de
# crochê (ponto pulado, tensao irregular, etc.) -- sao so 2 formas
# distintas de "quebrar" o padrao de linhas, para gerar variedade
# minima na classe "erro" e evitar que o mock seja trivial demais
# (uma unica variacao sempre igual).
TIPOS_ERRO = ["linha_deslocada", "linha_faltando"]

# Caminhos relativos ao proprio arquivo (nao ao diretorio de onde o
# script e chamado). Isso torna o script robusto a partir de onde ele
# e executado (ex: rodar de dentro de scripts/ ou da raiz do projeto
# da o mesmo resultado).
PASTA_RAIZ = os.path.join(os.path.dirname(__file__), "..")
PASTA_DATASET = os.path.join(PASTA_RAIZ, "dataset")
PASTA_HOLDOUT = os.path.join(PASTA_RAIZ, "dataset_teste_externo")

# Fixar as seeds de ambas as bibliotecas de aleatoriedade usadas
# (random do Python e numpy) -- cada uma tem seu proprio gerador
# interno, entao fixar so uma nao garante reprodutibilidade total.
random.seed(SEED)
np.random.seed(SEED)


def criar_pastas():
    """Garante que a estrutura de pastas dataset/{correto,erro} e
    dataset_teste_externo/{correto,erro} existe antes de gerar
    qualquer imagem. exist_ok=True evita erro caso as pastas ja
    tenham sido criadas manualmente antes (ex: passo do roadmap)."""
    for base in [PASTA_DATASET, PASTA_HOLDOUT]:
        for classe in ["correto", "erro"]:
            os.makedirs(os.path.join(base, classe), exist_ok=True)


def gerar_padrao_base(tamanho, espacamento, cor_fundo, cor_linha, espessura):
    """Gera uma imagem com linhas horizontais repetidas -- simula,
    de forma bem simplificada, o padrao de fileiras/carreiras
    regulares de um crochê bem feito. Nao tem nenhuma relacao visual
    real com croche -- e so uma estrutura repetitiva e previsivel,
    que o modelo pode aprender a reconhecer como "padrao normal"."""
    img = np.full((tamanho, tamanho, 3), cor_fundo, dtype=np.uint8)
    y = espacamento
    while y < tamanho:
        cv2.line(img, (0, y), (tamanho, y), cor_linha, espessura)
        y += espacamento
    return img


def gerar_imagem_correto(tamanho):
    """Classe 'correto': padrao de linhas uniforme, sem nenhuma
    quebra. Cores de fundo/linha e espacamento variam aleatoriamente
    dentro de faixas pequenas -- isso evita que todas as imagens
    'corretas' sejam identicas pixel a pixel, o que tornaria o
    problema trivial demais (o modelo so precisaria decorar uma
    imagem, nao aprender um padrao)."""
    espacamento = random.randint(28, 34)
    espessura = random.randint(2, 4)
    cor_fundo = tuple(int(c) for c in np.random.randint(180, 220, size=3))
    cor_linha = tuple(int(c) for c in np.random.randint(60, 100, size=3))
    return gerar_padrao_base(tamanho, espacamento, cor_fundo, cor_linha, espessura)


def gerar_imagem_erro(tamanho, tipo_erro):
    """Classe 'erro': mesmo padrao de linhas da classe 'correto', mas
    com UMA linha "estragada" de forma aleatoria -- simula uma
    anomalia pontual no meio de um padrao normal, analogo a um
    defeito localizado numa peca de crochê real."""
    espacamento = random.randint(28, 34)
    espessura = random.randint(2, 4)
    cor_fundo = tuple(int(c) for c in np.random.randint(180, 220, size=3))
    cor_linha = tuple(int(c) for c in np.random.randint(60, 100, size=3))

    img = np.full((tamanho, tamanho, 3), cor_fundo, dtype=np.uint8)
    linhas_y = list(range(espacamento, tamanho, espacamento))

    # Escolhe UMA linha aleatoria (indice, nao posicao em pixels)
    # para aplicar o defeito -- garante que o erro nao esteja sempre
    # no mesmo lugar da imagem, o que ensinaria o modelo a "olhar
    # sempre no mesmo canto" em vez de aprender o conceito de anomalia.
    idx_defeito = random.randint(0, len(linhas_y) - 1)

    for i, y in enumerate(linhas_y):
        if i == idx_defeito:
            if tipo_erro == "linha_faltando":
                # Simula "ponto pulado": a linha simplesmente nao e
                # desenhada, criando uma lacuna no padrao.
                continue
            elif tipo_erro == "linha_deslocada":
                # Simula "tensao irregular"/"carreira torta": a linha
                # e desenhada, mas deslocada verticalmente em relacao
                # a posicao esperada.
                y_deslocado = y + random.choice([-12, -8, 8, 12])
                cv2.line(img, (0, y_deslocado), (tamanho, y_deslocado), cor_linha, espessura)
                continue
        cv2.line(img, (0, y), (tamanho, y), cor_linha, espessura)

    return img


def gerar_lote(pasta_base, qtd_por_classe, prefixo_extra, csv_writer):
    """Gera um lote balanceado (mesma quantidade de 'correto' e
    'erro') e registra cada arquivo gerado no CSV de rotulagem.

    O parametro prefixo_extra existe para diferenciar os nomes de
    arquivo do lote principal (dataset/) dos nomes do lote de holdout
    (dataset_teste_externo/), evitando colisao de nomes caso alguem
    um dia junte os dois lotes numa mesma pasta.
    """
    for i in range(1, qtd_por_classe + 1):
        # --- imagem da classe "correto" ---
        nome_correto = f"correto_{prefixo_extra}{i:02d}.jpg"
        img_correto = gerar_imagem_correto(RESOLUCAO)
        cv2.imwrite(os.path.join(pasta_base, "correto", nome_correto), img_correto)
        csv_writer.writerow([nome_correto, "correto", "-"])

        # --- imagem da classe "erro" ---
        # Alterna entre os tipos de erro disponiveis (i % len(...))
        # para garantir que ambos os tipos apareçam distribuidos ao
        # longo do lote, em vez de gerar só um tipo de defeito.
        tipo = TIPOS_ERRO[i % len(TIPOS_ERRO)]
        nome_erro = f"erro_{prefixo_extra}{i:02d}.jpg"
        img_erro = gerar_imagem_erro(RESOLUCAO, tipo)
        cv2.imwrite(os.path.join(pasta_base, "erro", nome_erro), img_erro)
        csv_writer.writerow([nome_erro, "erro", tipo])


def main():
    criar_pastas()

    # CSV principal (dataset de treino/validacao/teste): espelha a
    # planilha de rotulagem descrita no PROJETO_CROCHE_IA.md (secao 6),
    # com as colunas Arquivo / Classe / Tipo de erro. Isso permite que
    # o pipeline de carregamento (proximo notebook) já aprenda a ler
    # os rotulos a partir de um CSV, nao so varrendo nomes de pasta --
    # o que vai ser exatamente o formato usado com as fotos reais.
    caminho_csv = os.path.join(PASTA_DATASET, "labels_mock.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Arquivo", "Classe", "Tipo de erro"])
        gerar_lote(PASTA_DATASET, QTD_TREINO_POR_CLASSE, "", writer)

    # CSV do holdout (lote de teste externo, nunca visto pelo modelo
    # durante treino/validacao/teste).
    caminho_csv_holdout = os.path.join(PASTA_HOLDOUT, "labels_mock_holdout.csv")
    with open(caminho_csv_holdout, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Arquivo", "Classe", "Tipo de erro"])
        gerar_lote(PASTA_HOLDOUT, QTD_HOLDOUT_POR_CLASSE, "extra_", writer)

    print("Mock gerado com sucesso.")
    print(f"- {QTD_TREINO_POR_CLASSE * 2} imagens em: {PASTA_DATASET}")
    print(f"- {QTD_HOLDOUT_POR_CLASSE * 2} imagens de holdout em: {PASTA_HOLDOUT}")
    print(f"- CSV principal: {caminho_csv}")
    print(f"- CSV holdout: {caminho_csv_holdout}")


if __name__ == "__main__":
    main()