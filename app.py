import pandas as pd
import numpy as np

# dataset
url_dataset = "series_data.csv"

try:
    datos = pd.read_csv(url_dataset)

    print("--- CARGA DE DATOS ---")
    print(f"Total de ítems cargados: {len(datos)}")
    print(f"Columnas del dataset: {list(datos.columns)}")
except Exception as e:
    print(f"Error crítico al cargar: {e}")

print("\n--- TRATAMIENTO DE DATOS ---")
print(f"Filas antes de la limpieza: {len(datos)}")

# 1. Eliminar duplicados
datos = datos.drop_duplicates()

# 2. Tratar nulos en la descripción (la columna real es 'Overview')
datos['Overview'] = datos['Overview'].fillna('Sin descripción disponible')

# 3. Asegurar tipos de datos (la columna real es 'IMDB_Rating')
datos['IMDB_Rating'] = datos['IMDB_Rating'].astype(float)

print(f"Filas después de la limpieza: {len(datos)}")

# ALTA (Create)
def añadir_serie(df, titulo, genero, descripcion, valoracion):
    # Error a probar: campo obligatorio vacío
    if not titulo or not genero:
        return df, "Error: El título y el género son obligatorios."

    # Evitar el error de len(datos) usando el max() + 1
    nuevo_id = int(df['id_serie'].max() + 1) if not df.empty else 1

    nueva_fila = pd.DataFrame({
        'id_serie': [nuevo_id],
        'titulo': [titulo],
        'genero': [genero],
        'descripcion': [descripcion],
        'valoracion': [valoracion]
    })
    return pd.concat([df, nueva_fila], ignore_index=True), f"Éxito: '{titulo}' añadida con ID {nuevo_id}."


# CONSULTA (Read) - Actualizada para incluir búsqueda por título
def consultar_serie(df, id_serie=None, genero=None, titulo=None):
    if id_serie is not None:
        resultado = df[df['id_serie'] == id_serie]
        if resultado.empty:
            return f"Error: No existe la serie con ID {id_serie}."
        return resultado
    elif titulo is not None and titulo.strip() != "":
        # Busca coincidencias parciales sin distinguir mayúsculas/minúsculas
        resultado = df[df['titulo'].str.contains(titulo, case=False, na=False)]
        if resultado.empty:
            return f"Error: No existe ninguna serie que contenga '{titulo}'."
        return resultado
    elif genero is not None and genero.strip() != "":
        return df[df['genero'].str.contains(genero, case=False, na=False)]
    
    return "Error: Indica un ID, un título o un género."


# MODIFICACIÓN (Update)
def modificar_serie(df, id_serie, columna, nuevo_valor):
    # Error a probar: id inexistente
    if id_serie not in df['id_serie'].values:
        return df, f"Error: No se puede modificar. El ID {id_serie} no existe."
    # Error a probar: columna inválida
    if columna not in df.columns:
        return df, f"Error: La columna '{columna}' no existe."

    # Usar .loc para evitar el SettingWithCopyWarning
    df.loc[df['id_serie'] == id_serie, columna] = nuevo_valor
    return df, f"Éxito: Serie {id_serie} actualizada ({columna} -> {nuevo_valor})."


# BAJA (Delete)
def eliminar_serie(df, id_serie):
    # Error a probar: id inexistente
    if id_serie not in df['id_serie'].values:
        return df, f"Error: No se puede eliminar. El ID {id_serie} no existe."

    df = df[df['id_serie'] != id_serie]
    # Requisito explícito: reset_index para evitar desalineaciones
    df = df.reset_index(drop=True)
    return df, f"Éxito: Serie con ID {id_serie} eliminada."

# Antes de probar, debemos adaptar el dataset original para que tenga las columnas esperadas por las funciones CRUD
if 'id_serie' not in datos.columns:
    # Crear un ID secuencial para cada serie
    datos['id_serie'] = datos.index + 1

# Renombrar las columnas originales en inglés a los nombres en español requeridos
columnas_a_renombrar = {
    'Series_Title': 'titulo',
    'Genre': 'genero',
    'Overview': 'descripcion',
    'IMDB_Rating': 'valoracion'
}

# Renombrar solo las columnas que existan en el DataFrame
datos = datos.rename(columns={k: v for k, v in columnas_a_renombrar.items() if k in datos.columns})

print("--- 1. PRUEBAS DE ALTA ---")
# Caso correcto
datos, msj = añadir_serie(datos, "The Office", "Comedia", "Documental falso", 9.0)
print(msj)
# Caso de error
_, msj_err = añadir_serie(datos, "", "Comedia", "Sin título", 0)
print(msj_err)

print("\n--- 2. PRUEBAS DE CONSULTA ---")
# Caso correcto
print(consultar_serie(datos, id_serie=1))
# Caso de error
print(consultar_serie(datos, id_serie=999999))

print("\n--- 3. PRUEBAS DE MODIFICACIÓN ---")
# Caso correcto
datos, msj = modificar_serie(datos, 1, 'valoracion', 9.9)
print(msj)
# Caso de error
datos, msj_err = modificar_serie(datos, 999999, 'valoracion', 10)
print(msj_err)

print("\n--- 4. PRUEBAS DE BAJA ---")
# Caso correcto
datos, msj = eliminar_serie(datos, 2)
print(msj)
print("Comprobación visual de que no está la serie 2:")
print(datos[['id_serie', 'titulo']].head())
# Caso de error
datos, msj_err = eliminar_serie(datos, 999999)
print(msj_err)

# ==========================================
# MOTOR DE RECOMENDACIÓN (TF-IDF)
# ==========================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def recomendar(titulo_buscado, n=5):
    global datos
    
    # 1. Seguridad: Comprobar que el título existe
    if titulo_buscado not in datos['titulo'].values:
        return pd.DataFrame({"Mensaje": [f"No se encontró '{titulo_buscado}' en el catálogo."]})
        
    # 2. Limpieza básica de la columna descripción
    datos['descripcion'] = datos['descripcion'].fillna("")
    
    # 3. Vectorización TF-IDF (Convierte el texto en matemáticas)
    tfidf = TfidfVectorizer(stop_words='english')
    matriz_tfidf = tfidf.fit_transform(datos['descripcion'])
    
    # 4. Cálculo de la similitud del coseno entre todas las series
    similitud = cosine_similarity(matriz_tfidf, matriz_tfidf)
    
    # 5. Obtener el índice numérico de la serie que buscó el usuario
    idx = datos[datos['titulo'] == titulo_buscado].index[0]
    
    # 6. Ordenar las series más similares (de mayor a menor afinidad)
    puntajes_similitud = list(enumerate(similitud[idx]))
    puntajes_similitud = sorted(puntajes_similitud, key=lambda x: x[1], reverse=True)
    
    # 7. Extraer los índices de las 'n' mejores (saltando la primera, que es la misma serie)
    indices_recomendados = [i[0] for i in puntajes_similitud[1:n+1]]
    
    # 8. Devolver las columnas clave para mostrar en la tabla de Gradio
    return datos.iloc[indices_recomendados][['titulo', 'genero', 'descripcion', 'valoracion']]

# Prueba rápida en consola para verificar que funciona antes de abrir Gradio
print(recomendar("Game of Thrones", 3))

# ==========================================
# SPRINT 3 - PROTOTIPO GRADIO (VERSIÓN FINAL)
# ==========================================
import gradio as gr
import pandas as pd

titulos_series = sorted(datos['titulo'].dropna().unique())

# Función auxiliar para Modificar y Eliminar (que siguen necesitando el ID internamente)
def obtener_id(entrada):
    if not entrada or entrada.strip() == "": return None
    if entrada.strip().isdigit(): return int(entrada)
    coincidencias = datos[datos['titulo'].str.contains(entrada, case=False, na=False)]
    if not coincidencias.empty: return int(coincidencias.iloc[0]['id_serie'])
    return None

# ---------------------------------------------------------
# 1. WRAPPERS PARA LA PESTAÑA 1 (CRUD)
# ---------------------------------------------------------
def wrapper_añadir(titulo, genero, descripcion, valoracion):
    global datos
    datos, msj = añadir_serie(datos, titulo, genero, descripcion, valoracion)
    return datos.tail(), msj

def wrapper_consultar(entrada_busqueda, genero):
    # Detectamos si la entrada es un número (ID) o texto (Título)
    es_numero = entrada_busqueda.strip().isdigit()
    
    id_val = int(entrada_busqueda) if es_numero else None
    titulo_val = None if es_numero else entrada_busqueda
    genero_val = genero if genero.strip() != "" else None
    
    # Llamamos a tu función actualizada aprovechando el nuevo parámetro 'titulo'
    resultado = consultar_serie(datos, id_serie=id_val, titulo=titulo_val, genero=genero_val)
    
    if isinstance(resultado, str): 
        return pd.DataFrame(), resultado
    return resultado, "Consulta realizada con éxito."

def wrapper_modificar(entrada_busqueda, columna, nuevo_valor):
    global datos
    id_serie = obtener_id(entrada_busqueda)
    if id_serie is None: return pd.DataFrame(), f"Error: No se encontró la serie '{entrada_busqueda}'."
    if columna == 'valoracion':
        try: nuevo_valor = float(nuevo_valor)
        except ValueError: pass
    datos, msj = modificar_serie(datos, id_serie, columna, nuevo_valor)
    return datos[datos['id_serie'] == id_serie], msj

def wrapper_eliminar(entrada_busqueda):
    global datos
    id_serie = obtener_id(entrada_busqueda)
    if id_serie is None: return pd.DataFrame(), f"Error: No se encontró la serie '{entrada_busqueda}'."
    datos, msj = eliminar_serie(datos, id_serie)
    return datos.tail(), msj

# ---------------------------------------------------------
# 2. WRAPPER PARA LA PESTAÑA 2: RECOMENDACIONES
# ---------------------------------------------------------
def wrapper_recomendar(titulo_seleccionado, n):
    return recomendar(titulo_seleccionado, int(n))

# ---------------------------------------------------------
# 3. WRAPPERS PARA LA PESTAÑA 3: FUNCIONES SIMULADAS
# ---------------------------------------------------------
def wrapper_valorar_simulado(titulo, estrellas):
    return f"(simulado) Has valorado '{titulo}' con {estrellas} estrellas."

def wrapper_wishlist_simulada(titulo):
    return f"(simulado) '{titulo}' se ha añadido a tu Wishlist correctamente."

def wrapper_compatibilidad_simulada():
    return "(simulado) Calculando afinidad con tus seguidores... ¡Tenéis un 85% de compatibilidad!"

# ==========================================
# CONSTRUCCIÓN DE LA INTERFAZ VISUAL
# ==========================================
with gr.Blocks(title="WatchWise - Prototipo", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎬 WatchWise - Prototipo Sprint 3")
    
    with gr.Tab("Base de Datos (Catálogo)"):
        gr.Markdown("Gestiona el catálogo de series. Las operaciones modifican la base de datos en tiempo real.")
        
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
                    
                gr.Markdown("#### Modificar campo (requiere buscar la serie arriba)")
                in_columna = gr.Dropdown(choices=['titulo', 'genero', 'descripcion', 'valoracion'], label="Columna a modificar")
                in_nuevo_valor = gr.Textbox(label="Nuevo Valor")
                btn_modificar = gr.Button("Aplicar Modificación")
                
        gr.Markdown("### Resultados de la Operación")
        out_mensaje_crud = gr.Textbox(label="Estado del Sistema", interactive=False)
        out_tabla_crud = gr.Dataframe(label="Datos de la Serie")

        btn_añadir.click(fn=wrapper_añadir, inputs=[in_titulo, in_genero, in_desc, in_val], outputs=[out_tabla_crud, out_mensaje_crud])
        btn_consultar.click(fn=wrapper_consultar, inputs=[in_busqueda, in_gen_buscar], outputs=[out_tabla_crud, out_mensaje_crud])
        btn_modificar.click(fn=wrapper_modificar, inputs=[in_busqueda, in_columna, in_nuevo_valor], outputs=[out_tabla_crud, out_mensaje_crud])
        btn_eliminar.click(fn=wrapper_eliminar, inputs=[in_busqueda], outputs=[out_tabla_crud, out_mensaje_crud])

    with gr.Tab("Descubrir (Recomendaciones)"):
        gr.Markdown("Selecciona una serie de nuestro catálogo y descubre opciones similares basadas en la sinopsis.")
        with gr.Row():
            entrada_titulo = gr.Dropdown(choices=titulos_series, label="Serie que te gustó")
            entrada_n = gr.Slider(minimum=3, maximum=20, value=5, step=1, label="Número de recomendaciones")
        boton_recomendar = gr.Button("Obtener Recomendaciones", variant="primary")
        salida_tabla = gr.Dataframe(label="Series Recomendadas")
        boton_recomendar.click(fn=wrapper_recomendar, inputs=[entrada_titulo, entrada_n], outputs=salida_tabla)

    with gr.Tab("Diario y Social (Simulado)"):
        gr.Markdown("Interactúa con la comunidad. Estas funciones representan la visión futura de WatchWise [Nivel Simulado].")
        with gr.Row():
            with gr.Column():
                social_titulo = gr.Dropdown(choices=titulos_series, label="Selecciona una Serie")
                social_estrellas = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Tu valoración")
                with gr.Row():
                    btn_valorar = gr.Button("Valorar (simulado)")
                    btn_wishlist = gr.Button("Añadir a Wishlist (simulado)")
            with gr.Column():
                gr.Markdown("<br>")
                btn_compatibilidad = gr.Button("Calcular Compatibilidad con Amigos (simulado)", variant="primary")
                
        out_mensaje_social = gr.Textbox(label="Respuesta del Sistema", interactive=False)
        btn_valorar.click(fn=wrapper_valorar_simulado, inputs=[social_titulo, social_estrellas], outputs=out_mensaje_social)
        btn_wishlist.click(fn=wrapper_wishlist_simulada, inputs=[social_titulo], outputs=out_mensaje_social)
        btn_compatibilidad.click(fn=wrapper_compatibilidad_simulada, inputs=[], outputs=out_mensaje_social)

app.launch(server_name="0.0.0.0", server_port=10000)