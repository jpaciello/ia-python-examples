import pandas as pd

# Mensaje para ambientes sin acceso al módulo `openai`
def responder_query_mock(query, datos):
    return (
        "El entorno actual no tiene acceso al módulo `openai`. Para ejecutar correctamente este script, asegúrate de instalar el paquete `openai` en tu entorno Python. "
        "Puedes hacerlo ejecutando: `pip install openai`. Mientras tanto, puedes utilizar este mensaje simulado."
    )

# Cargar el archivo de datos desde un directorio local
archivo_datos = r"amazon_reduced.csv"
data = pd.read_csv(archivo_datos, quotechar='"')

# Función para procesar la consulta del usuario y devolver una respuesta
def query_chatgpt(query, datos):
    try:
        # Verificar si el módulo `openai` está disponible
        import openai

        # Configurar la clave de la API de OpenAI
        # https://platform.openai.com/api-keys
        openai.api_key = "API KEY DE OPENAI"

        # Convertir todo el DataFrame a un formato legible
        datos_completos = datos.to_dict()

        # Crear el contexto para ChatGPT
        prompt = (
            "El archivo de datos de Amazon contiene información sobre ventas. A continuación, se proporcionan los datos:\n"
            f"{datos_completos}\n\n"
            "Utiliza estos datos para responder a la siguiente consulta del usuario de manera estadística y completa: \n"
            f"Query: {query}\n"
            "Respuesta:"
        )

        # Llamada a la API de ChatGPT usando openai.ChatCompletion.create
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Cambiar a "gpt-4" si es necesario
            messages=[
                {"role": "system", "content": "Eres un asistente que ayuda con análisis de datos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )

        return respuesta["choices"][0]["message"]["content"]
    except ModuleNotFoundError:
        return responder_query_mock(query, datos)
    except Exception as e:
        return f"Ocurrió un error al procesar la consulta: {e}"

# Ejemplo de uso
if __name__ == "__main__":
    print("Cargando datos...")
    try:
        print("Datos cargados correctamente. Por favor, ingresa tu consulta.")
        consulta_usuario = input("Ingresa tu consulta sobre los datos de Amazon: ")
        respuesta = query_chatgpt(consulta_usuario, data)
        print("\nRespuesta:")
        print(respuesta)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
