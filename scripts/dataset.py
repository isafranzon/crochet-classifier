"""
Classe Dataset customizada, compartilhada entre os notebooks de treino
(04_treino.ipynb) e avaliacao (05_avaliacao.ipynb).

Le as imagens a partir de um CSV (train.csv, val.csv, ou o CSV de
holdout), em vez de varrer uma pasta inteira como o ImageFolder faria --
isso permite respeitar exatamente a divisao treino/validacao/teste
definida no notebook 03, e usar transformacoes do albumentations.
"""

import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


class CrochetDataset(Dataset):
    """Dataset customizado que le as imagens listadas em um CSV de
    rotulagem (colunas: Arquivo, Classe, Tipo de erro).
    """

    def __init__(self, caminho_csv, pasta_imagens, transform=None):
        self.df = pd.read_csv(caminho_csv)
        self.pasta_imagens = pasta_imagens
        self.transform = transform

        # Mapeamento fixo de classe para indice numerico -- precisa ser
        # o MESMO mapeamento em treino, validacao e teste, senao o
        # modelo aprenderia (ou seria avaliado com) rotulos invertidos.
        self.classe_para_indice = {"correto": 0, "erro": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        nome_arquivo = linha["Arquivo"]
        classe = linha["Classe"]

        # Descobre a subpasta (correto/ ou erro/) a partir da classe,
        # ja que o CSV nao guarda o caminho completo, so o nome do arquivo.
        caminho_imagem = os.path.join(self.pasta_imagens, classe, nome_arquivo)

        # OpenCV le em BGR; convertendo para RGB para manter consistencia
        # com os demais notebooks do projeto.
        imagem = cv2.imread(caminho_imagem)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        if self.transform:
            resultado = self.transform(image=imagem)
            imagem = resultado["image"]

        rotulo = self.classe_para_indice[classe]
        return imagem, rotulo