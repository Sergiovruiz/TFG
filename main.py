# main.py
import os

from Codigos.carga_config import leer_configuracion
from Codigos.procesado_pdf import extraer_texto_pdf
from Codigos.carga_modelo import cargar_modelo, ejecutar_modelo, extraer_json
from Codigos.database import conectar_mongo, guardar_resultado
from Codigos.grid import crear_matriz
from Codigos.utils import construir_prompt, guardar_json


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config.txt")
PDF_BASE_PATH = os.path.join(BASE_DIR, "pdfs")
SALIDAS_DIR = os.path.join(BASE_DIR, "Salidas")

# Crear carpeta Salidas si no existe
os.makedirs(SALIDAS_DIR, exist_ok=True)

# Leer configuración
modelos, documentos, temperaturas, prompts = leer_configuracion(CONFIG_PATH)

# Convertir documentos a rutas absolutas dentro de la carpeta /pdfs
documentos = [os.path.join(PDF_BASE_PATH, doc) for doc in documentos]

print("\nCONFIGURACIÓN CARGADA")
print("Modelos:", modelos)
print("Documentos:", documentos)
print("Temperaturas:", temperaturas)
print(f"Prompts detectados: {len(prompts)}")


# Conectar a MongoDB
coleccion = conectar_mongo()

# Construir matriz de combinaciones
matriz = crear_matriz(modelos, documentos, temperaturas, prompts)

print("\nEJECUTANDO MATRIZ DE COMBINACIONES")
print("Total combinaciones:", len(matriz))


# Ejecución global
for modelo, pdf, temperatura, prompt_base in matriz:

    print(f"\n Ejecutando combinación:")
    print(f"   Modelo: {modelo}")
    print(f"   PDF: {pdf}")
    print(f"   Temperatura: {temperatura}")
    print(f"   Prompt ID: {prompts.index(prompt_base)+1}")

    # Cargar modelo
    llm = cargar_modelo(modelo)

    # Leer PDF
    texto_pdf = extraer_texto_pdf(pdf)

    # Construir prompt final
    prompt_final = construir_prompt(prompt_base, texto_pdf)

    # Ejecutar modelo
    raw_output = ejecutar_modelo(llm, prompt_final, float(temperatura))

    # Extraer JSON
    data = extraer_json(raw_output)

    # Guardar JSON en carpeta Salidas/
    indice_prompt = prompts.index(prompt_base) + 1
    salida = guardar_json(data,modelo,pdf,temperatura,indice_prompt,SALIDAS_DIR)
    print(f"JSON guardado en: {salida}")

    # Guardar en MongoDB
    guardar_resultado(coleccion,modelo,pdf,temperatura,indice_prompt,prompt_base,data)

print("\nPROCESO COMPLETADO")
