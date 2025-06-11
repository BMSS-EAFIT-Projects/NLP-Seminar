# %%
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import math


# %% [markdown]
# # Lectura de documentos

# %%
archivo = r"C:\Users\afpue\Documents\Seminario NLP\NLP\publicaciones\binaria.xlsx"
df = pd.read_excel(archivo)

def limpiar(texto):
    texto = texto.lower()
    return re.findall(r'\b\w+\b', texto)

docs = df['Documento'].apply(limpiar).tolist()
clases = df['Clase'].tolist()


# %%
nuevo_tweet = "covid es una pandemia mundial que afecta a todos"
nuevo_procesado = limpiar(nuevo_tweet)

# %% [markdown]
# # Bayes original

# %%
def train_naive_bayes(D, C):
    """
    Entrena un clasificador Multinomial Naive Bayes con add-1 smoothing.

    Parámetros:
    - D: lista de documentos (listas de palabras)
    - C: lista de clases

    Retorna:
    - V: vocabulario (lista de palabras únicas)
    - logprior: log P(c)
    - loglikelihood: log P(w|c)
    """
    N_doc = len(D)
    logprior = {}
    loglikelihood = {}
    bigdoc = defaultdict(list)
    vocab = set()

    for doc, c in zip(D, C):
        bigdoc[c].extend(doc)
        vocab.update(doc)
    
    V = list(vocab)

    for c in bigdoc:
        Nc = C.count(c)
        logprior[c] = math.log(Nc / N_doc)

        word_counts = Counter(bigdoc[c])
        denom = sum(word_counts[w] + 1 for w in V)  # denominador con add-1

        for w in V:
            num = word_counts[w] + 1
            loglikelihood[(w, c)] = math.log(num / denom)

    return V, logprior, loglikelihood

# %%
def test_naive_bayes(testdoc, logprior, loglikelihood, C, V, verbose=True):
    """
    Clasifica un documento con el modelo entrenado.

    Parámetros:
    - testdoc: lista de palabras
    - logprior, loglikelihood: parámetros entrenados
    - C: lista de clases posibles
    - V: vocabulario
    - verbose: si True, imprime las probabilidades logarítmicas

    Retorna:
    - clase_predicha: clase con mayor score
    """
    sum_scores = {}
    V_set = set(V)

    for c in C:
        score = logprior[c]
        for word in testdoc:
            if word in V_set:
                score += loglikelihood.get((word, c), 0.0)
        sum_scores[c] = score

    if verbose:
        print("Log-probabilidades por clase:")
        for c, score in sum_scores.items():
            print(f"  {c}: {score:.4f}")

    return max(sum_scores, key=sum_scores.get)

# %%
V, logprior, loglikelihood = train_naive_bayes(docs, clases)

prediccion = test_naive_bayes(nuevo_procesado, logprior, loglikelihood, list(set(clases)), V)

print("Predicción:", prediccion)


