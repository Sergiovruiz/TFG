from llama_cpp import Llama
import fitz
import json
import re
from pymongo import MongoClient
import os
import numpy as np



# ================================================================
# FUNCIÓN PARA LEER EL ARCHIVO DE CONFIGURACIÓN (MÚLTIPLES PROMPTS)
# ================================================================
def leer_configuracion(ruta_config):
    modelos = []
    documentos = []
    temperaturas = []
    prompts = []

    seccion_actual = None
    leyendo_prompt = False
    prompt_buffer = []

    if not os.path.exists(ruta_config):
        print(f"ERROR: No se encontró el archivo de configuración {ruta_config}")
        exit()

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
            if linea == "[TEMPERATURA]":
                seccion_actual = "temperatura"
                continue
            if linea == "[PROMPTS]":
                seccion_actual = "prompts"
                continue

            # Inicio de un nuevo prompt
            if seccion_actual == "prompts" and linea.startswith("##"):
                # Si había un prompt sin cerrar (raro pero seguro)
                if prompt_buffer:
                    prompts.append("\n".join(prompt_buffer))
                    prompt_buffer = []
                continue

            if linea == "FIN_PROMPT":
                prompts.append("\n".join(prompt_buffer))
                prompt_buffer = []
                leyendo_prompt = False
                continue

            # Capturar líneas del prompt
            if seccion_actual == "prompts":
                prompt_buffer.append(linea)
                leyendo_prompt = True
                continue

            # Secciones normales
            if seccion_actual == "modelos":
                modelos.append(linea.replace("\\", "/"))
            elif seccion_actual == "documentos":
                documentos.append(linea.replace("\\", "/"))
            elif seccion_actual == "temperatura":
                try:
                    temperaturas.append(float(linea))
                except:
                    print("Temperatura inválida, ignorada.")

    if not temperaturas:
        temperaturas = [0.6]

    if not prompts:
        print("ADVERTENCIA: No se encontraron prompts. Usando un prompt vacío.")
        prompts = [""]

    return modelos, documentos, temperaturas, prompts


# ================================================================
# LEER CONFIGURACIÓN
# ================================================================
ruta_config = "F:/Git/TFG/config.txt"
modelos, documentos, temperaturas, prompts = leer_configuracion(ruta_config)

print("\n=== MODELOS DETECTADOS ===")
for m in modelos:
    print(" -", m)

print("\n=== DOCUMENTOS DETECTADOS ===")
for d in documentos:
    print(" -", d)

print("\n=== TEMPERATURAS DETECTADAS ===")
for t in temperaturas:
    print(" -", t)

print("\n=== PROMPTS DETECTADOS ===")
for i, p in enumerate(prompts):
    print(f"\n--- PROMPT #{i+1} (preview) ---")
    print(p[:200], "...")


# ================================================================
# CONEXIÓN CON MONGODB
# ================================================================
client = MongoClient("mongodb://localhost:27017/")
db = client["analisis_pdfs"]
coleccion = db["resumenes"]


# ================================================================
# FUNCIÓN PRINCIPAL DE PROCESADO
# ================================================================
def analizar_pdf_con_modelo(ruta_modelo, ruta_pdf, temperatura, prompt_base):

    print(f"\nCargando modelo: {ruta_modelo}")
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

    # Leer PDF
    try:
        doc = fitz.open(ruta_pdf)
    except:
        print(f"ERROR al abrir PDF: {ruta_pdf}")
        return None

    texto = ""
    for pagina in doc:
        texto += pagina.get_text("text") + "\n"
    doc.close()

    if not texto.strip():
        print("PDF vacío o sin texto.")
        return None

    # Construir prompt final
    prompt = f"""{prompt_base}

Texto a analizar:
\"\"\"{texto[:4000]}\"\"\"
Asistente:
"""

    # Inferencia
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

    try:
        data = json.loads(json_limpio)
    except:
        data = {"respuesta_raw": respuesta}

    # Guardar archivo
    nom_modelo = os.path.basename(ruta_modelo).replace(".gguf", "")
    nom_pdf = os.path.basename(ruta_pdf).replace(".pdf","")

    # Añadir índice del prompt
    idx_prompt = prompts.index(prompt_base) + 1
    nombre_salida = f"resumen_{nom_modelo}_{nom_pdf}_T{temperatura}_P{idx_prompt}.json"

    with open(nombre_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON guardado: {nombre_salida}")

    # MongoDB
    coleccion.insert_one({
        "archivo_pdf": ruta_pdf,
        "modelo": ruta_modelo,
        "temperatura": temperatura,
        "prompt_id": idx_prompt,
        "prompt_texto": prompt_base,
        "resultado": data
    })

    return data


# ================================================================
#        CREAR MATRIZ DE COMBINACIONES (MODELO × DOC × TEMP × PROMPT)
# ================================================================
print("\n🔧 Construyendo matriz completa de combinaciones...\n")

grid = np.array(np.meshgrid(modelos, documentos, temperaturas, prompts)).T.reshape(-1, 4)

print("📌 TOTAL COMBINACIONES:", len(grid))
for m, d, t, p in grid:
    print(f" Modelo: {m} | Doc: {d} | Temp: {t} | Prompt: {prompts.index(p)+1}")


# ================================================================
#                     EJECUCIÓN GLOBAL
# ================================================================
print("\n🚀 Ejecutando matriz completa de pruebas...\n")

for mdl, doc, temp, prompt in grid:
    analizar_pdf_con_modelo(mdl, doc, float(temp), prompt)

print("\n🎯 PROCESO COMPLETADO — MATRIZ 4D EJECUTADA\n")
