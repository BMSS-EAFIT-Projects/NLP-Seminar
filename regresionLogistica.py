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
archivo = r"C:\Users\afpue\Documents\Seminario NLP\NLP\publicaciones\binaria.xlsx"
df = pd.read_excel(archivo)
# Asegurar que las clases sean 0 (negativo) y 1 (positivo)
df['Clase'] = df['Clase'].map({'negativo': 0, 'positivo': 1})


# %% [markdown]
# Funciones

# %%
# --- Función para limpiar datos ---
def limpiar(texto):
    texto = texto.lower()
    return re.findall(r'\b\w+\b', texto)

# --- Función sigmoide ---
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# --- Producto punto ---
def dot(w, x):
    return sum(w_i * x_i for w_i, x_i in zip(w, x))

# --- Predicción de probabilidad ---
def predict_proba(w, x):
    return sigmoid(dot(w, x))

# --- Pérdida logística (log-loss) promedio ---
def log_loss(w, X, y):
    loss = 0.0
    for xi, yi in zip(X, y):
        p = predict_proba(w, xi)
        p = max(min(p, 1 - 1e-15), 1e-15)  # para evitar log(0)
        loss += -yi * math.log(p) - (1 - yi) * math.log(1 - p)
    return loss / len(y)

# --- Gradiente del log-loss ---
def compute_gradient(w, X, y):
    grad = [0.0] * len(w)
    n = len(y)
    for xi, yi in zip(X, y):
        p = predict_proba(w, xi)
        for j in range(len(w)):
            grad[j] += (p - yi) * xi[j]
    return [g / n for g in grad]

# --- Entrenamiento con descenso de gradiente ---
def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    w = [random.uniform(-0.01, 0.01) for _ in range(len(X[0]))]
    for epoch in range(epochs):
        grad = compute_gradient(w, X, y)
        w = [w_i - lr * g_i for w_i, g_i in zip(w, grad)]
        if epoch % 100 == 0 or epoch == epochs - 1:
            loss = log_loss(w, X, y)
            print(f"Época {epoch}: pérdida = {loss:.4f}")
    return w

# --- Predicción final binaria ---
def predict(w, x, threshold=0.5):
    return 1 if predict_proba(w, x) >= threshold else 0

# %% [markdown]
# Caracteristicas / Features

# %%
palabras_positivas = {"feliz", "alegre", "contento", "maravilloso", "excelente", "genial", "bueno", "positivo", "fantástico", "me encanta"}
palabras_negativas = {"triste", "deprimido", "mal", "horrible", "terrible", "enfermo", "negativo", "odio", "me molesta", "estresado", 
                      "muertes", "muerte", "enfermedad", "dolor", "sufrimiento", "tragedia", "desastre", "crisis", "problema", "conflicto",
                      }



# %%
def extraer_caracteristicas(tweet):
    tokens = limpiar(tweet)
    x0 = 1
    x1 = sum(1 for palabra in tokens if palabra in palabras_positivas)
    x2 = sum(1 for palabra in tokens if palabra in palabras_negativas)
    x3 = len(tokens)
    return [x0, x1, x2, x3]


# %% [markdown]
# Datos

# %%
# X será una lista de listas con las características
X = [extraer_caracteristicas(t) for t in df['Documento']]
# y será una lista con las clases: 0 = negativo, 1 = positivo
y = df['Clase'].tolist()


# %% [markdown]
# Entrenamiento

# %%
w = train_logistic_regression(X, y, lr=0.01, epochs=1000)


# %% [markdown]
# Prediccion

# %%
nuevo_tweet = "Me siento muy feliz y positivo hoy"
x_nuevo = extraer_caracteristicas(nuevo_tweet)
prediccion = predict(w, x_nuevo)

print("Predicción:", prediccion)  # 1 si positivo, 0 si negativo


# %%
import math
import re
import random

# === Definiciones ===
palabras_felices = {"feliz", "alegre", "contento", "maravilloso", "excelente", "genial", "bueno", "positivo", "fantástico", "encanta"}
palabras_tristes = {"triste", "deprimido", "mal", "horrible", "terrible", "enfermo", "negativo", "odio", "molesta", "estresado"}

def limpiar(texto):
    texto = str(texto).lower()
    return re.findall(r'\b\w+\b', texto)

def extraer_caracteristicas(tweet):
    tokens = limpiar(tweet)
    x0 = 1
    x1 = sum(1 for palabra in tokens if palabra in palabras_felices)
    x2 = sum(1 for palabra in tokens if palabra in palabras_tristes)
    x3 = len(tokens)
    return [x0, x1, x2, x3]

def sigmoid(z):
    if z < -700:
        return 0.0
    elif z > 700:
        return 1.0
    return 1 / (1 + math.exp(-z))

def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def predict_proba(w, x):
    return sigmoid(dot(w, x))

def log_loss(w, X, y):
    loss = 0.0
    for xi, yi in zip(X, y):
        p = predict_proba(w, xi)
        p = max(min(p, 1 - 1e-15), 1e-15)
        loss += -yi * math.log(p) - (1 - yi) * math.log(1 - p)
    return loss / len(y)

def compute_gradient(w, X, y):
    grad = [0.0] * len(w)
    n = len(y)
    for xi, yi in zip(X, y):
        p = predict_proba(w, xi)
        for j in range(len(w)):
            grad[j] += (p - yi) * xi[j]
    return [g / n for g in grad]

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    w = [random.uniform(-0.01, 0.01) for _ in range(len(X[0]))]
    for epoch in range(epochs):
        grad = compute_gradient(w, X, y)
        w = [wi - lr * gi for wi, gi in zip(w, grad)]
        if epoch % 100 == 0 or epoch == epochs - 1:
            loss = log_loss(w, X, y)
            print(f"Época {epoch}: pérdida = {loss:.4f}")
    return w

def predict(w, x, threshold=0.5):
    return 1 if predict_proba(w, x) >= threshold else 0


# %%
import pandas as pd

# Leer archivo
archivo = r"C:\Users\afpue\Documents\Seminario NLP\NLP\publicaciones\binaria.xlsx"
df = pd.read_excel(archivo)

# Asegurar codificación binaria de clase
df['Clase'] = df['Clase'].map({'Negativo': 0, 'Positivo': 1})

# Último tweet como test
tweet_test = df.iloc[-1]['Documento']
y_test = df.iloc[-1]['Clase']
x_test = extraer_caracteristicas(tweet_test)

# Resto como entrenamiento
df_train = df.iloc[:-1]
X_train = [extraer_caracteristicas(texto) for texto in df_train['Documento']]
y_train = df_train['Clase'].tolist()


# %%
w = train_logistic_regression(X_train, y_train, lr=0.01, epochs=1000)

# Predicción final
proba = predict_proba(w, x_test)
clase_predicha = predict(w, x_test)

print("\n--- Resultado final ---")
print("Tweet:", tweet_test)
print("Clase real:", y_test)
print("Clase predicha:", clase_predicha)
print(f"Probabilidad estimada de clase positiva: {proba:.4f}")



