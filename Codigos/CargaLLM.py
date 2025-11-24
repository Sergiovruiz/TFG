from llama_cpp import Llama
import fitz
import json
import re
from pymongo import MongoClient
import os


# Funcion para leer el archivo de configuración
def leer_configuracion(ruta_config):
    modelos = []
    documentos = []
    temperatura = 0.6
    prompt = ""

    if not os.path.exists(ruta_config):
        print(f"ERROR: No se encontró el archivo de configuración: {ruta_config}")
        exit()

    seccion_actual = None
    leyendo_prompt = False
    prompt_buffer = []

    with open(ruta_config, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()

            # Ignorar comentarios o líneas vacías
            if not linea or linea.startswith("#"):
                continue

            if linea == "[MODELOS]":
                seccion_actual = "modelos"
                continue
            if linea == "[DOCUMENTOS]":
                seccion_actual = "documentos"
                continue
            if linea == "[TEMPERATURA]":
                seccion_actual = "temperatura"
                continue
            if linea == "[PROMPT]":
                seccion_actual = "prompt"
                leyendo_prompt = True
                continue
            if linea == "FIN_PROMPT":
                leyendo_prompt = False
                continue

            if leyendo_prompt:
                prompt_buffer.append(linea)
                continue

            if seccion_actual == "modelos":
                modelos.append(linea.replace("\\", "/"))

            elif seccion_actual == "documentos":
                documentos.append(linea.replace("\\", "/"))

            elif seccion_actual == "temperatura":
                try:
                    temperatura = float(linea)
                except ValueError:
                    print("Temperatura inválida, usando valor por defecto 0.6")
                    temperatura = 0.6

    prompt = "\n".join(prompt_buffer)
    return modelos, documentos, temperatura, prompt


ruta_config = "F:/Git/TFG/config.txt"
modelos, documentos, temperatura, prompt_base = leer_configuracion(ruta_config)

print("\nMODELOS DETECTADOS")
for m in modelos:
    print(" -", m)

print("\nDOCUMENTOS DETECTADOS")
for d in documentos:
    print(" -", d)

print(f"\nTEMPERATURA: {temperatura}")
print("\nPROMPT CARGADO (primeros 200 chars):")
print(prompt_base[:200], "...")


#Conexion con MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["analisis_pdfs"]
coleccion = db["resumenes"]


# Funcion de procesado de PDF
def analizar_pdf_con_modelo(ruta_modelo, ruta_pdf):
    print(f"\nCargando modelo:\n{ruta_modelo}\n")
    llm = Llama(
        model_path=ruta_modelo,
        n_ctx=2048,
        n_threads=12,
        n_gpu_layers=30,
        n_batch=512,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )

    print(f"Procesando PDF: {ruta_pdf}")

    # Extraer texto
    try:
        doc = fitz.open(ruta_pdf)
    except:
        print(f"Error al abrir el PDF: {ruta_pdf}")
        return None

    texto = ""
    for pagina in doc:
        texto += pagina.get_text("text") + "\n"
    doc.close()

    if not texto.strip():
        print("No se ha encontrado texto en el PDF.")
        return None

    # Construir prompt
    prompt = f"""{prompt_base}

Texto a analizar:
\"\"\"{texto[:4000]}\"\"\"
Asistente:
"""

    # Ejecutar modelo
    output = llm(
        prompt,
        max_tokens=700,
        temperature=temperatura,
        top_p=0.9,
        stop=["Usuario:", "Asistente:"],
        echo=False
    )

    respuesta = output["choices"][0]["text"].strip()

    # Extraer JSON
    bloques_json = re.findall(r"\{[\s\S]*\}", respuesta)
    json_limpio = bloques_json[-1] if bloques_json else respuesta

    json_limpio = json_limpio.replace("<|end|>", "").strip()

    # Respuesta -> JSON
    try:
        data = json.loads(json_limpio)
    except json.JSONDecodeError:
        print("JSON inválido. Guardando salida RAW.")
        data = {"respuesta_raw": respuesta}

    # Guardar archivo
    nombre_modelo = os.path.basename(ruta_modelo).replace(".gguf", "")
    nombre_pdf = os.path.basename(ruta_pdf).replace(".pdf", "")

    nombre_salida = f"resumen_{nombre_modelo}_{nombre_pdf}.json"

    with open(nombre_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON guardado en {nombre_salida}")

    # Guardar en MongoDB
    coleccion.insert_one({
        "archivo_pdf": ruta_pdf,
        "modelo": ruta_modelo,
        "temperatura": temperatura,
        "resultado": data
    })

    print("Guardado en MongoDB")

    return data


# Ejecucion en bucle (TODO: ejecucion basada en matriz de combinaciones)
for modelo in modelos:
    for documento in documentos:
        analizar_pdf_con_modelo(modelo, documento)

print("\n PROCESO COMPLETADO\n")
