# %% [markdown]
# Librerias

# %%
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import math
from sklearn.model_selection import KFold

# %% [markdown]
# Archivos de entrenamiento

# %%
archivo = r"C:\Users\afpue\Documents\Seminario NLP\NLP\publicaciones\binaria.xlsx"
df = pd.read_excel(archivo)

def limpiar(texto):
    texto = texto.lower()
    return re.findall(r'\b\w+\b', texto)

docs = df['Documento'].apply(limpiar).tolist()
clases = df['Clase'].tolist()


# %% [markdown]
# Tweet de prueba

# %%
nuevo_procesado = limpiar(df.iloc[-1]['Documento'])
clase_real = df.iloc[-1]['Clase']

docs = df.iloc[:-1]['Documento'].apply(limpiar).tolist()
clases = df.iloc[:-1]['Clase'].tolist()

# %% [markdown]
# # Naive Bayes

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
        denom = sum(word_counts[w] + 1 for w in V)

        for w in V:
            num = word_counts[w] + 1
            loglikelihood[(w, c)] = math.log(num / denom)

    return V, logprior, loglikelihood

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

# %% [markdown]
# # Cross validation de naive Bayes

# %%
def cross_validate_naive_bayes(docs, clases, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    total_aciertos = 0
    total_instancias = 0
    porcentajes = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(docs), 1):
        D_train = [docs[i] for i in train_index]
        C_train = [clases[i] for i in train_index]
        D_test = [docs[i] for i in test_index]
        C_test = [clases[i] for i in test_index]

        V, logprior, loglikelihood = train_naive_bayes(D_train, C_train)

        aciertos_fold = 0
        for doc, true_class in zip(D_test, C_test):
            pred = test_naive_bayes(doc, logprior, loglikelihood, list(set(C_train)), V, verbose=False)
            if pred == true_class:
                aciertos_fold += 1

        porcentaje_fold = aciertos_fold / len(D_test)
        porcentajes.append(porcentaje_fold)
        total_aciertos += aciertos_fold
        total_instancias += len(D_test)

        print(f"Fold {fold_idx}: {porcentaje_fold:.2%} de acierto ({aciertos_fold}/{len(D_test)})")

    exactitud_promedio = total_aciertos / total_instancias
    print(f"\nExactitud promedio en {k}-fold cross-validation: {exactitud_promedio:.4f}")


# %%
docs = df['Documento'].apply(limpiar).tolist()
clases = df['Clase'].tolist()

cross_validate_naive_bayes(docs, clases, k=5)

# %% [markdown]
# # Binary naive Bayes

# %%
def train_binary_naive_bayes(D, C):
    """
    Entrena un clasificador Binary Multinomial Naive Bayes con add-1 smoothing.
    En este modelo solo se considera si una palabra aparece al menos una vez por documento.

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
    
    bin_docs = [list(set(doc)) for doc in D]

    for doc, c in zip(bin_docs, C):
        bigdoc[c].extend(doc)
        vocab.update(doc)
    
    V = list(vocab)
    
    for c in bigdoc:
        Nc = C.count(c)
        logprior[c] = math.log(Nc / N_doc)

        word_counts = Counter(bigdoc[c])
        denom = sum(word_counts[w] + 1 for w in V)  # add-1 smoothing

        for w in V:
            num = word_counts[w] + 1
            loglikelihood[(w, c)] = math.log(num / denom)

    return V, logprior, loglikelihood

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

# %% [markdown]
# # Cross validation de binary naive Bayes

# %%
def cross_validate_binary_naive_bayes(docs, clases, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    total_aciertos = 0
    total_instancias = 0
    porcentajes = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(docs), 1):
        D_train = [docs[i] for i in train_index]
        C_train = [clases[i] for i in train_index]
        D_test = [docs[i] for i in test_index]
        C_test = [clases[i] for i in test_index]

        V, logprior, loglikelihood = train_binary_naive_bayes(D_train, C_train)

        aciertos_fold = 0
        for doc, true_class in zip(D_test, C_test):
            doc_bin = list(set(doc)) 
            pred = test_naive_bayes(doc_bin, logprior, loglikelihood, list(set(C_train)), V, verbose=False)
            if pred == true_class:
                aciertos_fold += 1

        porcentaje_fold = aciertos_fold / len(D_test)
        porcentajes.append(porcentaje_fold)
        total_aciertos += aciertos_fold
        total_instancias += len(D_test)

        print(f"Fold {fold_idx}: {porcentaje_fold:.2%} de acierto ({aciertos_fold}/{len(D_test)})")

    exactitud_promedio = total_aciertos / total_instancias
    print(f"\nExactitud promedio en {k}-fold cross-validation: {exactitud_promedio:.4f}")


# %%
docs = df['Documento'].apply(limpiar).tolist()
clases = df['Clase'].tolist()

cross_validate_binary_naive_bayes(docs, clases, k=5)


