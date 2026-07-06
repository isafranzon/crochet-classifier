"""
Gerador de imagens sinteticas (mock) para testar o pipeline
de classificacao correto/erro sem depender das fotos reais.

NAO ensina nada sobre crochê de verdade — so replica a mesma
estrutura de pastas e formato de arquivo das fotos reais.
"""

import os
import csv
import random
import numpy as np
import cv2

# ---------- CONFIGURACOES ----------
RESOLUCAO = 512          # imagem "crua", resize fica por conta do pipeline depois
QTD_TREINO_POR_CLASSE = 20   # 20 correto + 20 erro = 40 no total
QTD_HOLDOUT_POR_CLASSE = 3   # 3 correto + 3 erro = 6 extras (teste 5 do doc principal)
SEED = 42

TIPOS_ERRO = ["linha_deslocada", "linha_faltando"]

PASTA_RAIZ = os.path.join(os.path.dirname(__file__), "..")
PASTA_DATASET = os.path.join(PASTA_RAIZ, "dataset")
PASTA_HOLDOUT = os.path.join(PASTA_RAIZ, "dataset_teste_externo")

random.seed(SEED)
np.random.seed(SEED)


def criar_pastas():
    for base in [PASTA_DATASET, PASTA_HOLDOUT]:
        for classe in ["correto", "erro"]:
            os.makedirs(os.path.join(base, classe), exist_ok=True)


def gerar_padrao_base(tamanho, espacamento, cor_fundo, cor_linha, espessura):
    """Gera uma imagem com linhas horizontais repetidas — simula
    o padrao de pontos de crochê em fileiras regulares."""
    img = np.full((tamanho, tamanho, 3), cor_fundo, dtype=np.uint8)
    y = espacamento
    while y < tamanho:
        cv2.line(img, (0, y), (tamanho, y), cor_linha, espessura)
        y += espacamento
    return img


def gerar_imagem_correto(tamanho):
    espacamento = random.randint(28, 34)
    espessura = random.randint(2, 4)
    cor_fundo = tuple(int(c) for c in np.random.randint(180, 220, size=3))
    cor_linha = tuple(int(c) for c in np.random.randint(60, 100, size=3))
    return gerar_padrao_base(tamanho, espacamento, cor_fundo, cor_linha, espessura)


def gerar_imagem_erro(tamanho, tipo_erro):
    espacamento = random.randint(28, 34)
    espessura = random.randint(2, 4)
    cor_fundo = tuple(int(c) for c in np.random.randint(180, 220, size=3))
    cor_linha = tuple(int(c) for c in np.random.randint(60, 100, size=3))

    img = np.full((tamanho, tamanho, 3), cor_fundo, dtype=np.uint8)
    linhas_y = list(range(espacamento, tamanho, espacamento))

    # escolhe uma linha aleatoria pra "estragar"
    idx_defeito = random.randint(0, len(linhas_y) - 1)

    for i, y in enumerate(linhas_y):
        if i == idx_defeito:
            if tipo_erro == "linha_faltando":
                continue  # pula essa linha -> "ponto pulado"
            elif tipo_erro == "linha_deslocada":
                y_deslocado = y + random.choice([-12, -8, 8, 12])
                cv2.line(img, (0, y_deslocado), (tamanho, y_deslocado), cor_linha, espessura)
                continue
        cv2.line(img, (0, y), (tamanho, y), cor_linha, espessura)

    return img


def gerar_lote(pasta_base, qtd_por_classe, prefixo_extra, csv_writer):
    for i in range(1, qtd_por_classe + 1):
        nome_correto = f"correto_{prefixo_extra}{i:02d}.jpg"
        img_correto = gerar_imagem_correto(RESOLUCAO)
        cv2.imwrite(os.path.join(pasta_base, "correto", nome_correto), img_correto)
        csv_writer.writerow([nome_correto, "correto", "-"])

        tipo = TIPOS_ERRO[i % len(TIPOS_ERRO)]
        nome_erro = f"erro_{prefixo_extra}{i:02d}.jpg"
        img_erro = gerar_imagem_erro(RESOLUCAO, tipo)
        cv2.imwrite(os.path.join(pasta_base, "erro", nome_erro), img_erro)
        csv_writer.writerow([nome_erro, "erro", tipo])


def main():
    criar_pastas()

    caminho_csv = os.path.join(PASTA_DATASET, "labels_mock.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Arquivo", "Classe", "Tipo de erro"])
        gerar_lote(PASTA_DATASET, QTD_TREINO_POR_CLASSE, "", writer)

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