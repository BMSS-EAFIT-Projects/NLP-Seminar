# %% [markdown]
# Librerias

# %%
import math
import random
import re
import pandas as pd

# %% [markdown]
# Datos

# %%
archivo = r"C:\Users\afpue\Documents\GitHub\NLP-Seminar\publicaciones\binaria_limpio.xlsx"
df = pd.read_excel(archivo)
df['Clase'] = df['Clase'].map({'Negativo': 0, 'Positivo': 1})

df['Documento'] = df['Documento_Lematizado']
def limpiar(texto):
    texto = texto.lower()
    return re.findall(r'\b\w+\b', texto)

# %% [markdown]
# Diccionarios

# %%
palabras_positivas = {"feliz", "alegre", "contento", "maravilloso", "excelente", "genial", "bueno", "positivo", "fantástico", "me encanta",
                      "filantropos", "filantropo"}
palabras_negativas = {"triste", "deprimido", "mal", "horrible", "terrible", "enfermo", "negativo", "odio", "me molesta", "estresado", 
                      "muertes", "muerte", "enfermedad", "dolor", "sufrimiento", "tragedia", "desastre", "crisis", "problema", "conflicto",
                      "infectados", "desgracia"}


# %% [markdown]
# Funciones

# %%
def extraer_caracteristicas(tweet, palabras_negativas, palabras_positivas):
    tokens = limpiar(tweet)
    x0 = 1  # Bias
    x1 = sum(1 for palabra in tokens if palabra in palabras_negativas)
    x2 = sum(1 for palabra in tokens if palabra in palabras_positivas)
    x3 = len(tokens)
    return [x0, x1, x2, x3]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def predict_proba(w, x):
    return sigmoid(dot(w, x))

def single_log_loss(p, y):
    p = max(min(p, 1 - 1e-15), 1e-15)
    return - y * math.log(p) - (1 - y) * math.log(1 - p)

def compute_batch_gradient(w, X_batch, y_batch):
    grad = [0.0 for _ in w]
    batch_size = len(y_batch)
    for xi, yi in zip(X_batch, y_batch):
        p = predict_proba(w, xi)
        for j in range(len(w)):
            grad[j] += (p - yi) * xi[j]
    return [g / batch_size for g in grad]

def train_logistic_regression_minibatch(X, y, lr=0.01, epochs=1000, batch_size=10):
    w = [0.0 for _ in range(len(X[0]))]  # Inicializar pesos

    for epoch in range(epochs):
        combined = list(zip(X, y))
        random.shuffle(combined)

        # Dividir en mini-batches
        for i in range(0, len(y), batch_size):
            batch = combined[i:i + batch_size]
            X_batch, y_batch = zip(*batch)

            # Calcular gradiente promedio en el batch
            grad = compute_batch_gradient(w, X_batch, y_batch)

            # Actualizar pesos
            w = [wj - lr * gj for wj, gj in zip(w, grad)]

        # (Opcional) Calcular pérdida promedio en la época
        loss_epoch = sum(single_log_loss(predict_proba(w, xi), yi) for xi, yi in zip(X, y)) / len(y)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Época {epoch}: pérdida promedio = {loss_epoch:.4f}")

    return w

def predict(w, x, threshold=0.5):
    return 1 if predict_proba(w, x) >= threshold else 0


# %%
tweet_test = df.iloc[-1]['Documento']
y_test = df.iloc[-1]['Clase']
x_test = extraer_caracteristicas(tweet_test, palabras_negativas, palabras_positivas)

# Resto como entrenamiento
df_train = df.iloc[:-1]
X_train = [extraer_caracteristicas(texto, palabras_negativas, palabras_positivas) for texto in df_train['Documento']]
y_train = df_train['Clase'].tolist()


# %%
w = train_logistic_regression_minibatch(X_train, y_train, lr=0.01, epochs=1000, batch_size=5)

# Predicción final sobre un ejemplo de prueba
proba = predict_proba(w, x_test)
clase_predicha = predict(w, x_test)

print("\n--- Resultado final ---")
print("Tweet:", tweet_test)
print("Clase real:", y_test)
print("Clase predicha:", clase_predicha)
print(f"Probabilidad estimada de clase positiva: {proba:.4f}")


# %%
w


