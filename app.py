# -*- coding: utf-8 -*-
"""Sprint4ABP.ipynb

WatchWise - Sprint4: Prototipo Gradio (Versión Final con Pósters HTML)
"""

import os
import nltk
import numpy as np
import pandas as pd

# 1. CARGA DEL DATASET LIMPIO
url_dataset = "series_data_clean.csv"

try:
  datos = pd.read_csv(url_dataset)
  print("--- CARGA DE DATOS ---")
  print(f"Total de ítems cargados: {len(datos)}")
except Exception as e:
  print(f"Error crítico al cargar: {e}")

# Tratamiento de datos y nulos
datos = datos.drop_duplicates()
datos["Overview"] = datos["Overview"].fillna("Sin descripción disponible")
datos["IMDB_Rating"] = datos["IMDB_Rating"].astype(float)


# Función para transformar las URLs en etiquetas HTML seguras para Gradio
def convertir_a_html_img(url):
  if pd.isna(url) or not str(url).startswith("http"):
    return 'https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500'
  return f'<img src="{url}" width="55" style="border-radius: 4px; object-fit: cover;">'


# Aplicar la conversión HTML a los pósters
if "Poster_Link" in datos.columns:
  datos["Poster_Link"] = datos["Poster_Link"].apply(convertir_a_html_img)
else:
  datos["Poster_Link"] = (
      'https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500'
  )


# ALTA (Create)
def añadir_serie(df, titulo, genero, descripcion, valoracion):
  if not titulo or not genero:
    return df, "Error: El título y el género son obligatorios."

  nuevo_id = int(df["id_serie"].max() + 1) if "id_serie" in df.columns and not df.empty else len(df) + 1

  nueva_fila = pd.DataFrame({
      "id_serie": [nuevo_id],
      "titulo": [titulo],
      "genero": [genero],
      "descripcion": [descripcion],
      "valoracion": [valoracion],
      "Poster_Link": [
          '<img src="https://images.unsplash.com/photo-1485846234645-a62644f84728?w=500" width="55" style="border-radius: 4px;">'
      ],
  })
  return pd.concat([df, nueva_fila], ignore_index=True), (
      f"Éxito: '{titulo}' añadida con ID {nuevo_id}."
  )


# CONSULTA (Read)
def consultar_serie(df, id_serie=None, genero=None, titulo=None):
  if id_serie is not None:
    resultado = df[df["id_serie"] == id_serie]
    if resultado.empty:
      return f"Error: No existe la serie con ID {id_serie}."
    return resultado
  elif titulo is not None and titulo.strip() != "":
    resultado = df[df["titulo"].str.contains(titulo, case=False, na=False)]
    if resultado.empty:
      return f"Error: No existe ninguna serie que contenga '{titulo}'."
    return resultado
  elif genero is not None and genero.strip() != "":
    return df[df["genero"].str.contains(genero, case=False, na=False)]
  return "Error: Indica un ID, un título o un género."


# MODIFICACIÓN (Update)
def modificar_serie(df, id_serie, columna, nuevo_valor):
  if id_serie not in df["id_serie"].values:
    return df, f"Error: No se puede modificar. El ID {id_serie} no existe."
  if columna not in df.columns:
    return df, f"Error: La columna '{columna}' no existe."
  df.loc[df["id_serie"] == id_serie, columna] = nuevo_valor
  return df, f"Éxito: Serie {id_serie} actualizada ({columna} -> {nuevo_valor})."


# BAJA (Delete)
def eliminar_serie(df, id_serie):
  if id_serie not in df["id_serie"].values:
    return df, f"Error: No se puede eliminar. El ID {id_serie} no existe."
  df = df[df["id_serie"] != id_serie].reset_index(drop=True)
  return df, f"Éxito: Serie con ID {id_serie} eliminada."


if "id_serie" not in datos.columns:
  datos["id_serie"] = datos.index + 1

columnas_a_renombrar = {
    "Series_Title": "titulo",
    "Genre": "genero",
    "Overview": "descripcion",
    "IMDB_Rating": "valoracion",
}
datos = datos.rename(
    columns={
        k: v for k, v in columnas_a_renombrar.items() if k in datos.columns
    }
)


# ==========================================
# BLOQUE 1: RECOMENDADOR HÍBRIDO (TF-IDF + VADER)
# ==========================================
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

datos["descripcion"] = datos["descripcion"].fillna("")

tfidf = TfidfVectorizer(stop_words="english")
matriz_tfidf = tfidf.fit_transform(datos["descripcion"])
distancia_texto = pairwise_distances(matriz_tfidf, metric="cosine")

sia = SentimentIntensityAnalyzer()
datos["sentimiento"] = datos["descripcion"].apply(
    lambda t: sia.polarity_scores(str(t))["compound"]
)
distancia_sentimiento = pairwise_distances(
    datos[["sentimiento"]].values, metric="euclidean"
)
distancia_sentimiento /= distancia_sentimiento.max()


def recomendar_por_matriz(titulo_buscado, matriz_dist, n=5):
  if titulo_buscado not in datos["titulo"].values:
