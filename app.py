# -*- coding: utf-8 -*-
"""
WatchWise - Prototipo Sprint 4
Despliegue estable para Render con renderizado visual de pósters en Gradio
"""

import os
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn import svm
from sklearn.model_selection import train_test_split
import gradio as gr

# Descarga de recursos NLTK requeridos
nltk.download("vader_lexicon", quiet=True)
nltk.download("movie_reviews", quiet=True)

# 1. CARGA Y LIMPIEZA DE DATOS
url_dataset = "series_data_clean.csv"

try:
    datos = pd.read_csv(url_dataset)
    print("--- CARGA DE DATOS ---")
    print(f"Total de ítems cargados: {len(datos)}")
except Exception as e:
    print(f"Error al cargar dataset: {e}")
    datos = pd.DataFrame(columns=['id_serie', 'titulo', 'genero', 'descripcion', 'valoracion', 'Poster_Link'])

datos = datos.drop_duplicates()

# Normalización de nombres de columnas
columnas_a_renombrar = {
    'Series_Title': 'titulo',
    'Genre': 'genero',
    'Overview': 'descripcion',
    'IMDB_Rating': 'valoracion'
}
datos = datos.rename(columns={k: v for k, v in columnas_a_renombrar.items() if k in datos.columns})

# Tratamiento de nulos y tipos
if 'descripcion' in datos.columns:
    datos['descripcion'] = datos['descripcion'].fillna('Sin descripción disponible')
else:
    datos['descripcion'] = 'Sin descripción disponible'

if 'valoracion' in datos.columns:
    datos['valoracion'] = pd.to_numeric(datos['valoracion'], errors='coerce').fillna(0.0)
else:
    datos['valoracion'] = 0.0

if 'id_serie' not in datos.columns:
    datos['id_serie'] = datos.index + 1

# Transformación de URLs en elementos HTML img para renderizado en gr.Dataframe
def formatear_html_poster(url):
    url_str = str(url).strip()
    if not url_str.startswith('http') or url_str == 'nan':
        url_str = 'https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500'
    return f'<img src="{url_str}" width="55" style="border-radius:4px; object-fit:cover;">'

if 'Poster_Link' in datos.columns:
    datos['Poster_Link'] = datos['Poster_Link'].apply(formatear_html_poster)
else:
    datos['Poster_Link'] = '<img src="https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500" width="55" style="border-radius:4px; object-fit:cover;">'

# 2. FUNCIONES CRUD DE SERIES
def añadir_serie(df, titulo, genero, descripcion, valoracion):
    if not titulo or not genero:
        return df, "Error: El título y el género son obligatorios."

    nuevo_id = int(df['id_serie'].max() + 1) if not df.empty and 'id_serie' in df.columns else 1
    poster_defecto = '<img src="https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500" width="55" style="border-radius:4px; object-fit:cover;">'

    nueva_fila = pd.DataFrame([{
        'id_serie': nuevo_id,
        'Poster_Link': poster_defecto,
        'titulo': titulo,
        'genero': genero,
        'descripcion': descripcion,
        'valoracion': valoracion
    }])
    df_actualizado = pd.concat([df, nueva_fila], ignore_index=True)
    return df_actualizado, f"Éxito: '{titulo}' añadida con ID {nuevo_id}."

def consultar_serie(df, id_serie=None, genero=None, titulo=None):
    if id_serie is not None:
        res = df[df['id_serie'] == id_serie]
        return res if not res.empty else f"Error: No existe la serie con ID {id_serie}."
    elif titulo is not None and str(titulo).strip() != "":
        res = df[df['titulo'].str.contains(str(titulo), case=False, na=False)]
        return res if not res.empty else f"Error: No se encontró la serie '{titulo}'."
    elif genero is not None and str(genero).strip() != "":
        res = df[df['genero'].str.contains(str(genero), case=False, na=False)]
        return res if not res.empty else f"Error: No se encontraron series con el género '{genero}'."
    return "Error: Indica un ID, un título o un género."

def modificar_serie(df, id_serie, columna, nuevo_valor):
    if id_serie not in df['id_serie'].values:
        return df, f"Error: El ID {id_serie} no existe."
    if columna not in df.columns:
        return df, f"Error: La columna '{columna}' no existe."
    df.loc[df['id_serie'] == id_serie, columna] = nuevo_valor
    return df, f"Éxito: Serie {id_serie} actualizada ({columna} -> {nuevo_valor})."

def eliminar_serie(df, id_serie):
    if id_serie not in df['id_serie'].values:
        return df, f"Error: El ID {id_serie} no existe."
    df_actualizado = df[df['id_serie'] != id_serie].reset_index(drop=True)
    return df_actualizado, f"Éxito: Serie con ID {id_serie} eliminada."

# 3. MOTOR DE RECOMENDACIÓN HÍBRIDO (TF-IDF + VADER)
tfidf = TfidfVectorizer(stop_words='english')
matriz_tfidf = tfidf.fit_transform(datos['descripcion'])
distancia_texto = pairwise_distances(matriz_tfidf, metric="cosine")

sia = SentimentIntensityAnalyzer()
datos["sentimiento"] = datos["descripcion"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
distancia_sentimiento = pairwise_distances(datos[["sentimiento"]].values, metric="euclidean")
max_dist_sent = distancia_sentimiento.max()
if max_dist_sent > 0:
    distancia_sentimiento /= max_dist_sent

def recomendar_por_matriz(titulo_buscado, matriz_dist, n=5):
    if titulo_buscado not in datos['titulo'].values:
        return pd.DataFrame({"Mensaje": [f"No se encontró '{titulo_buscado}' en el catálogo."]})
    idx = datos[datos['titulo'] == titulo_buscado].index[0]
    orden = matriz_dist[idx].argsort()
    indices_rec = [i for i in orden if i != idx][:int(n)]
    columnas_mostrar = ['Poster_Link', 'titulo', 'genero', 'sentimiento', 'valoracion', 'descripcion']
    cols_existentes = [c for c in columnas_mostrar if c in datos.columns]
    return datos.iloc[indices_rec][cols_existentes]

# 4. CLASIFICADOR SVM Y VALORACIONES
datos_reviews = [
    (" ".join(movie_reviews.words(fileid)), categoria)
    for categoria in movie_reviews.categories()
    for fileid in movie_reviews.fileids(categoria)
]
df_reviews = pd.DataFrame(datos_reviews, columns=["texto", "categoria"])
df_reviews["valoracion"] = (df_reviews["categoria"] == "pos").astype(int)

X_train, X_resto, y_train, y_resto = train_test_split(
    df_reviews["texto"], df_reviews["valoracion"], test_size=0.4, stratify=df_reviews["valoracion"], random_state=42
)

vectorizador_svm = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizador_svm.fit_transform(X_train)

modelo_svm = svm.SVC(kernel="linear").fit(X_train_tfidf, y_train)

ETIQUETA = {1: "Like (1)", 0: "No Like (0)
