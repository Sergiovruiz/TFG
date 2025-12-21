# main.py
import os

from Codigos.carga_config import leer_configuracion
from Codigos.procesado_pdf import extraer_texto_pdf
from Codigos.carga_modelo import cargar_modelo, ejecutar_modelo, extraer_json
# from Codigos.database import conectar_mongo, guardar_resultado
from Codigos.grid import crear_matriz, crear_csv
from Codigos.utils import construir_prompt, guardar_json, escribir_en_csv


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(BASE_DIR, "config.txt")
PDF_BASE_PATH = os.path.join(BASE_DIR, "pdfs")
SALIDAS_DIR = os.path.join(BASE_DIR, "Salidas")
CSV_PATH = os.path.join(BASE_DIR, "Salidas", "combinaciones.csv")

# Crear carpeta Salidas si no existe
os.makedirs(SALIDAS_DIR, exist_ok=True)

# Leer configuración
modelos, documentos, temperaturas, prompts, repeticiones = leer_configuracion(CONFIG_PATH)

# Convertir documentos a rutas absolutas dentro de la carpeta /pdfs
documentos = [os.path.join(PDF_BASE_PATH, doc) for doc in documentos]

print("\nCONFIGURACIÓN CARGADA")
print("Modelos:", modelos)
print("Documentos:", documentos)
print("Temperaturas:", temperaturas)
print("Prompts detectados:", len(prompts))
print("Repeticiones:", repeticiones)

#Creacion de csv de resultados
crear_csv(modelos, documentos, temperaturas, prompts, repeticiones, CSV_PATH)

# Conectar a MongoDB
# coleccion = conectar_mongo()

# Construir matriz de combinaciones
matriz = crear_matriz(modelos, documentos, temperaturas, prompts)

print("\nEJECUTANDO MATRIZ DE COMBINACIONES")
print("Total combinaciones base:", len(matriz))
print("Total ejecuciones reales:", len(matriz) * repeticiones)

# Ejecución global
for modelo, pdf, temperatura, prompt_base in matriz:

    if prompt_base == "DEFAULT":
        indice_prompt = "Default"
    else:
        indice_prompt = prompts.index(prompt_base) + 1

    print(f"\n➡ Combinación:")
    print(f"   Modelo: {modelo}")
    print(f"   PDF: {pdf}")
    print(f"   Temperatura: {temperatura}")
    print(f"   Prompt ID: {indice_prompt}")

    # Cargar modelo UNA VEZ por combinación base
    llm = cargar_modelo(modelo)

    # Leer PDF UNA VEZ por combinación base
    texto_pdf = extraer_texto_pdf(pdf)

    for rep in range(1, repeticiones + 1):
        print(f"   Repetición {rep}/{repeticiones}")

        # Construir prompt, default si se esta ejecutando el prompt por defecto
        prompt_final = construir_prompt(texto_pdf, modelo, os.path.basename(pdf), temperatura, indice_prompt)

        # Ejecutar modelo
        raw_output = ejecutar_modelo(llm, prompt_final, float(temperatura))

        # Extraer JSON
        data = extraer_json(raw_output)

        # Guardar JSON (versionado automático)
        salida = guardar_json(data, modelo, pdf, temperatura, indice_prompt, SALIDAS_DIR)

        print(f"JSON guardado en: {salida}")

        metadata = data["metadata"]
        review = data["review"]

    escribir_en_csv(CSV_PATH, metadata, review, rep)
        
print("\nPROCESO COMPLETADO")
