from llama_cpp import Llama
import fitz  # PyMuPDF
import json
import re
from pymongo import MongoClient
import os


# FUNCIÓN PARA LEER ARCHIVO DE CONFIGURACIÓN
def leer_configuracion(ruta_config):
    modelos = []
    documentos = []

    if not os.path.exists(ruta_config):
        print(f"ERROR: No se encontró el archivo de configuración: {ruta_config}")
        exit()

    seccion_actual = None

    with open(ruta_config, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()

            if not linea or linea.startswith("#"):
                continue

            if linea == "[MODELOS]":
                seccion_actual = "modelos"
                continue

            if linea == "[DOCUMENTOS]":
                seccion_actual = "documentos"
                continue

            if seccion_actual == "modelos":
                modelos.append(linea)

            elif seccion_actual == "documentos":
                documentos.append(linea)

    return modelos, documentos


# LEER CONFIGURACIÓN
ruta_config = "F:\Git\TFG\config.txt"
modelos, documentos = leer_configuracion(ruta_config)

print("\nMODELOS DETECTADOS")
for m in modelos:
    print(" -", m)

print("\nDOCUMENTOS DETECTADOS")
for d in documentos:
    print(" -", d)


# CONEXIÓN A MONGODB
client = MongoClient("mongodb://localhost:27017/")
db = client["analisis_pdfs"]
coleccion = db["resumenes"]


# FUNCIÓN PARA PROCESAR UN PDF CON UN MODELO
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

    # EXTRAER TEXTO DEL PDF
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

    # PROMPT
    prompt = f"""
Eres un asistente que analiza documentos en español y devuelve exclusivamente un JSON con una estructura fija.

Instrucciones importantes:
- Devuelve SOLO el JSON.
- Respeta exactamente la estructura.
- Si un campo no existe, déjalo vacío.
- No añadas texto fuera del JSON.

Aquí está la estructura:

{{
  "documento": {{
    "titulo": "",
    "autor": "",
    "fecha": "",
    "resumen_general": "",
    "apartados": [
      {{
        "titulo": "Introducción",
        "contenido": "",
        "palabras_clave": []
      }},
      {{
        "titulo": "Objetivos",
        "contenido": "",
        "palabras_clave": []
      }},
      {{
        "titulo": "Metodología",
        "contenido": "",
        "palabras_clave": []
      }},
      {{
        "titulo": "Resultados",
        "contenido": "",
        "palabras_clave": []
      }},
      {{
        "titulo": "Conclusiones",
        "contenido": "",
        "palabras_clave": []
      }}
    ]
  }}
}}

Texto a analizar:
\"\"\"{texto[:4000]}\"\"\"
Asistente:
    """

    # EJECUTAR MODELO
    output = llm(
        prompt,
        max_tokens=700,
        temperature=0.6,
        top_p=0.9,
        stop=["Usuario:", "Asistente:"],
        echo=False
    )
    respuesta = output["choices"][0]["text"].strip()

    # EXTRAER JSON
    bloques_json = re.findall(r"\{[\s\S]*\}", respuesta)
    json_limpio = bloques_json[-1] if bloques_json else respuesta

    json_limpio = json_limpio.replace("<|end|>", "").strip()

    # PARSEO
    try:
        data = json.loads(json_limpio)
    except json.JSONDecodeError:
        print("JSON inválido. Guardando salida RAW.")
        data = {"respuesta_raw": respuesta}

    # GUARDAR ARCHIVO
    nombre_modelo = os.path.basename(ruta_modelo).replace(".gguf", "")
    nombre_pdf = os.path.basename(ruta_pdf).replace(".pdf", "")

    nombre_salida = f"resumen_{nombre_modelo}_{nombre_pdf}.json"

    with open(nombre_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON guardado en {nombre_salida}")
    
    # GUARDAR EN MONGODB
    coleccion.insert_one({
        "archivo_pdf": ruta_pdf,
        "modelo": ruta_modelo,
        "resultado": data
    })

    print("Guardado en MongoDB")

    return data


# === EJECUCIÓN GLOBAL ===
for modelo in modelos:
    for documento in documentos:
        analizar_pdf_con_modelo(modelo, documento)

print("\n=== PROCESO COMPLETADO ===\n")
