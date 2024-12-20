import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from sklearn.metrics.pairwise import cosine_similarity

# Define las rutas para los archivos CSV
file_path = os.path.join(os.path.dirname(__file__), 'GenerosListo.csv')  # Cambia esto a la ruta de tu dataset

# Cargar tu dataset de películas
df = pd.read_csv(file_path)

# Asegúrate de que los títulos en el DataFrame estén en minúsculas
df['title'] = df['title'].str.lower()

# Inicializa la aplicación FastAPI
app = FastAPI()

def recomendar_peliculas(titulo, df, num_recomendaciones=5):
    # Convertir el título ingresado a minúsculas
    titulo = titulo.lower()
    
    # Verificar si la película está en el DataFrame
    if titulo not in df['title'].values:
        return f"La película '{titulo}' no se encuentra en la base de datos."

    # Obtener la fila correspondiente a la película ingresada
    pelicula_input = df[df['title'] == titulo]
    
    # Calcular la similitud de coseno entre la película ingresada y todas las demás películas
    # Excluir las columnas que no son dummies (id, title, vote_average)
    dummies = df.drop(columns=['id', 'title', 'vote_average'])
    
    # Obtener el vector de características de la película ingresada
    vector_input = pelicula_input.drop(columns=['id', 'title', 'vote_average']).values
    
    # Calcular la similitud
    similitud = cosine_similarity(vector_input, dummies)[0]
    
    # Crear un DataFrame de similitudes
    df_similitud = pd.DataFrame({
        'title': df['title'],
        'similarity': similitud,
        'vote_average': df['vote_average']
    })
    
    # Filtrar y ordenar las películas por similitud, excluyendo la película de entrada
    df_similitud = df_similitud[df_similitud['title'] != titulo]
    
    # Primero, ordenamos por similitud en orden descendente
    df_similitud = df_similitud.sort_values(by='similarity', ascending=False)

    # Ahora, ordenamos las recomendaciones por vote_average en orden descendente
    recomendaciones = df_similitud.head(num_recomendaciones).sort_values(by='vote_average', ascending=False)

    # Devolver solo los títulos y el vote_average
    return recomendaciones[['title', 'vote_average']]

@app.get("/")
async def welcome(request: Request):
    # Obtener la URL base de la solicitud
    base_url = f"{request.url.scheme}://{request.url.hostname}" + (f":{request.url.port}" if request.url.port else "")
    
    return {
        "message": "Bienvenido a la API de recomendación de películas.",
        "funcionalidad": "Esta API te permite obtener recomendaciones de películas basadas en una película que ya conoces.",
        "ejemplo": {
            "url": f"{base_url}/recomendar/?title=Inception",
            "nota": "Reemplaza 'Inception' con el título de la película que conoces y te sugerirá 5 títulos similares. La cantidad de recomendaciones es fija y no puede ser modificada."
        }
    }

@app.get("/recomendar/")
async def recomendar_movies(title: str):
    recomendaciones = recomendar_peliculas(title, df)
    if isinstance(recomendaciones, str):  # Si la respuesta es un mensaje de error
        raise HTTPException(status_code=404, detail=recomendaciones)
    return {"recomendaciones": recomendaciones.to_dict(orient='records')}  # Devolver las recomendaciones como lista de diccionarios

# Para correr la aplicación, usa el siguiente comando en la terminal:
# uvicorn main:app --reload
