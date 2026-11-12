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

ETIQUETA = {1: "Like (1)", 0: "No Like (0)"}
df_valoraciones = pd.DataFrame(columns=["id_val", "usuario", "serie", "valoracion", "origen", "comentario"])

def guardar_valoracion(usuario, item, prediccion_num, origen="manual", texto=""):
    global df_valoraciones
    if not usuario or not str(usuario).strip() or not item:
        return df_valoraciones, "Error: El nombre de usuario y la serie son obligatorios."

    usuario = str(usuario).strip()
    etiqueta_str = ETIQUETA[int(prediccion_num)]

    filtro = (df_valoraciones["usuario"].str.lower() == usuario.lower()) & (df_valoraciones["serie"] == item)
    if filtro.any():
        df_valoraciones.loc[filtro, ["valoracion", "origen", "comentario"]] = [etiqueta_str, origen, texto]
        return df_valoraciones, f"Éxito: Valoración de '{item}' actualizada."

    nuevo_id = int(df_valoraciones["id_val"].max() + 1) if not df_valoraciones.empty else 1
    nueva = pd.DataFrame([{
        "id_val": nuevo_id,
        "usuario": usuario,
        "serie": item,
        "valoracion": etiqueta_str,
        "origen": origen,
        "comentario": texto
    }])
    df_valoraciones = pd.concat([df_valoraciones, nueva], ignore_index=True)
    return df_valoraciones, f"Éxito: Valoración guardada con ID {nuevo_id}."

def consultar_valoraciones(usuario_filtro=""):
    if not usuario_filtro or not str(usuario_filtro).strip():
        return df_valoraciones
    return df_valoraciones[df_valoraciones["usuario"].str.contains(str(usuario_filtro).strip(), case=False, na=False)]

def eliminar_valoracion(id_val):
    global df_valoraciones
    if id_val not in df_valoraciones["id_val"].values:
        return df_valoraciones, f"Error: No existe la valoración con ID {id_val}."
    df_valoraciones = df_valoraciones[df_valoraciones["id_val"] != id_val].reset_index(drop=True)
    return df_valoraciones, f"Éxito: Valoración con ID {id_val} eliminada."

def valoracion_automatica(comentario):
    vector = vectorizador_svm.transform([str(comentario).lower().strip()])
    return int(modelo_svm.predict(vector)[0])

# 5. WRAPPERS E INTERFAZ GRADIO
titulos_series = sorted(datos['titulo'].dropna().unique().tolist()) if 'titulo' in datos.columns else []
COLS_CATALOGO = ['Poster_Link', 'id_serie', 'titulo', 'genero', 'valoracion', 'descripcion']

def formatear_catalogo(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS_CATALOGO)
    cols = [c for c in COLS_CATALOGO if c in df.columns]
    return df[cols]

def obtener_id(entrada):
    if not entrada or str(entrada).strip() == "":
        return None
    if str(entrada).strip().isdigit():
        return int(entrada)
    coincidencias = datos[datos['titulo'].str.contains(str(entrada), case=False, na=False)]
    if not coincidencias.empty:
        return int(coincidencias.iloc[0]['id_serie'])
    return None

def wrapper_añadir(t, g, d, v):
    global datos
    datos, msj = añadir_serie(datos, t, g, d, v)
    return formatear_catalogo(datos.tail()), msj

def wrapper_consultar(entrada_busqueda, genero):
    if (not entrada_busqueda or str(entrada_busqueda).strip() == "") and (not genero or str(genero).strip() == ""):
        return formatear_catalogo(datos), "Mostrando el catálogo completo."

    es_numero = str(entrada_busqueda).strip().isdigit()
    id_val = int(entrada_busqueda) if es_numero else None
    titulo_val = None if es_numero else entrada_busqueda
    genero_val = genero if str(genero).strip() != "" else None
    resultado = consultar_serie(datos, id_serie=id_val, titulo=titulo_val, genero=genero_val)

    if isinstance(resultado, str):
        return pd.DataFrame(columns=COLS_CATALOGO), resultado
    return formatear_catalogo(resultado), "Consulta realizada con éxito."

def wrapper_modificar(entrada_busqueda, col, val):
    global datos
    id_s = obtener_id(entrada_busqueda)
    if id_s is None:
        return pd.DataFrame(columns=COLS_CATALOGO), f"Error: No se encontró la serie '{entrada_busqueda}'."
    if col == 'valoracion':
        try:
            val = float(val)
        except ValueError:
            pass
    datos, msj = modificar_serie(datos, id_s, col, val)
    return formatear_catalogo(datos[datos['id_serie'] == id_s]), msj

def wrapper_eliminar(entrada_busqueda):
    global datos
    id_s = obtener_id(entrada_busqueda)
    if id_s is None:
        return pd.DataFrame(columns=COLS_CATALOGO), f"Error: No se encontró la serie '{entrada_busqueda}'."
    datos, msj = eliminar_serie(datos, id_s)
    return formatear_catalogo(datos.tail()), msj

def recomendar_ui(tit, alpha, n):
    matriz = float(alpha) * distancia_texto + (1.0 - float(alpha)) * distancia_sentimiento
    return recomendar_por_matriz(tit, matriz, int(n))

def valorar_manual_ui(usr, itm, dec, com):
    pred = 1 if dec == "Like" else 0
    tabla, msj = guardar_valoracion(usr, itm, pred, origen="manual", texto=com)
    return msj, tabla

def valorar_auto_ui(usr, itm, com):
    if not com or not str(com).strip():
        return "Error: Escribe una reseña para que el modelo analice el sentimiento.", consultar_valoraciones()
    pred = valoracion_automatica(com)
    tabla, msj = guardar_valoracion(usr, itm, pred, origen="automática", texto=com)
    return f"Predicción SVM: {ETIQUETA[pred]} | {msj}", tabla

def filtrar_val_ui(usr):
    return consultar_valoraciones(usr)

def eliminar_val_ui(id_v):
    if id_v is None:
        return "Error: Indica un ID válido.", consultar_valoraciones()
    tabla, msj = eliminar_valoracion(int(id_v))
    return msj, tabla

with gr.Blocks(title="WatchWise - Prototipo Sprint 4") as app:
    gr.Markdown("# 🎬 WatchWise - Prototipo Sprint 4")

    with gr.Tab("Base de Datos (Catálogo)"):
        gr.Markdown("Gestiona el catálogo de series en tiempo real.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Añadir Nueva Serie")
                in_titulo = gr.Textbox(label="Título")
                in_genero = gr.Textbox(label="Género")
                in_desc = gr.Textbox(label="Descripción")
                in_val = gr.Number(label="Valoración Inicial", value=0.0)
                btn_añadir = gr.Button("Añadir Serie", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Consultar / Modificar / Eliminar")
                in_busqueda = gr.Textbox(label="ID o Título de la Serie")
                in_gen_buscar = gr.Textbox(label="O buscar por Género")
                with gr.Row():
                    btn_consultar = gr.Button("Buscar")
                    btn_eliminar = gr.Button("Eliminar", variant="stop")
                gr.Markdown("#### Modificar campo")
                in_columna = gr.Dropdown(choices=['titulo', 'genero', 'descripcion', 'valoracion'], label="Columna a modificar")
                in_nuevo_valor = gr.Textbox(label="Nuevo Valor")
                btn_modificar = gr.Button("Aplicar Modificación")

        gr.Markdown("### Resultados de la Operación")
        out_mensaje_crud = gr.Textbox(label="Estado del Sistema", interactive=False)
        out_tabla_crud = gr.Dataframe(
            label="Datos de la Serie",
            datatype=["html", "num", "str", "str", "num", "str"]
        )

        btn_añadir.click(fn=wrapper_añadir, inputs=[in_titulo, in_genero, in_desc, in_val], outputs=[out_tabla_crud, out_mensaje_crud])
        btn_consultar.click(fn=wrapper_consultar, inputs=[in_busqueda, in_gen_buscar], outputs=[out_tabla_crud, out_mensaje_crud])
        btn_modificar.click(fn=wrapper_modificar, inputs=[in_busqueda, in_columna, in_nuevo_valor], outputs=[out_tabla_crud, out_mensaje_crud])
        btn_eliminar.click(fn=wrapper_eliminar, inputs=[in_busqueda], outputs=[out_tabla_crud, out_mensaje_crud])

    with gr.Tab("Descubrir (Recomendaciones)"):
        gr.Markdown("Combina similitud de contenido (TF-IDF) y sentimiento (VADER).")
        with gr.Row():
            entrada_titulo = gr.Dropdown(choices=titulos_series, value=titulos_series[0] if titulos_series else None, label="Serie que te gustó")
            entrada_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Peso contenido (alpha)")
            entrada_n = gr.Slider(minimum=3, maximum=20, value=5, step=1, label="Nº de recomendaciones")
        boton_recomendar = gr.Button("Obtener Recomendaciones", variant="primary")
        salida_tabla = gr.Dataframe(
            label="Series Recomendadas",
            datatype=["html", "str", "str", "num", "num", "str"]
        )
        boton_recomendar.click(fn=recomendar_ui, inputs=[entrada_titulo, entrada_alpha, entrada_n], outputs=salida_tabla)

    with gr.Tab("Valorar y Social"):
        gr.Markdown("Subsistema de Valoración **[Manual CRUD + Automática SVM]**.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Valorar Serie")
                social_usuario = gr.Textbox(label="Usuario", value="Ana")
                social_titulo = gr.Dropdown(choices=titulos_series, value=titulos_series[0] if titulos_series else None, label="Selecciona Serie")
                social_decision = gr.Radio(choices=["Like", "No Like"], value="Like", label="Valoración Manual")
                social_comentario = gr.Textbox(label="Reseña (en inglés para SVM)", lines=2, placeholder="Great plot and amazing characters!")
                with gr.Row():
                    btn_val_manual = gr.Button("Guardar Valoración Manual", variant="primary")
                    btn_val_auto = gr.Button("Predecir Automática (SVM)")
            with gr.Column(scale=1):
                gr.Markdown("### Gestión de Valoraciones")
                filtro_usr = gr.Textbox(label="Filtrar por Usuario")
                btn_consultar_val = gr.Button("Actualizar Tabla")
                id_borrar_val = gr.Number(label="ID de Valoración a eliminar", value=1, precision=0)
                btn_eliminar_val = gr.Button("Eliminar Valoración", variant="stop")

        out_mensaje_social = gr.Textbox(label="Respuesta del Sistema", interactive=False)
        out_tabla_val = gr.Dataframe(
            label="Base de Datos de Valoraciones",
            value=consultar_valoraciones()
        )

        btn_val_manual.click(fn=valorar_manual_ui, inputs=[social_usuario, social_titulo, social_decision, social_comentario], outputs=[out_mensaje_social, out_tabla_val])
        btn_val_auto.click(fn=valorar_auto_ui, inputs=[social_usuario, social_titulo, social_comentario], outputs=[out_mensaje_social, out_tabla_val])
        btn_consultar_val.click(fn=filtrar_val_ui, inputs=[filtro_usr], outputs=out_tabla_val)
        btn_eliminar_val.click(fn=eliminar_val_ui, inputs=[id_borrar_val], outputs=[out_mensaje_social, out_tabla_val])

port = int(os.environ.get("PORT", 10000))
app.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(), share=False)
