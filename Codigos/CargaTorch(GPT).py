from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import fitz  # PyMuPDF
import json
import re
from pymongo import MongoClient
import os


# Este codigo ha sido generado por ChatGPT. La intencion es utilizarlo como
# comparativa para valorar si seria buena idea utilizar torch frente a LlamaCpp.


# ---------------------------------------------
# FUNCIÓN PARA LEER ARCHIVO DE CONFIGURACIÓN
# ---------------------------------------------
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
    buffer_prompt = []

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

            if linea == "[PROMPT]":
                leyendo_prompt = True
                continue

            if linea == "FIN_PROMPT":
                leyendo_prompt = False
                continue

            # Prompt multilínea
            if leyendo_prompt:
                buffer_prompt.append(linea)
                continue

            # Secciones normales
            if seccion_actual == "modelos":
                modelos.append(linea.replace("\\", "/"))

            elif seccion_actual == "documentos":
                documentos.append(linea.replace("\\", "/"))

            elif seccion_actual == "temperatura":
                try:
                    temperatura = float(linea)
                except:
                    print("Temperatura no válida, usando 0.6")

    return modelos, documentos, temperatura, "\n".join(buffer_prompt)


# ---------------------------------------------
# CARGA DEL MODELO (PyTorch)
# ---------------------------------------------
def cargar_modelo_torch(ruta_modelo):
    print(f"\n📌 Cargando modelo PyTorch: {ruta_modelo}\n")

    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)

    model = AutoModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.float16,
        device_map="auto"     # Usa GPU si existe
    )

    return tokenizer, model


# ---------------------------------------------
# GENERACIÓN DE TEXTO
# ---------------------------------------------
def generar_con_torch(tokenizer, model, prompt, temperatura):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=700,
        temperature=temperatura,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------------------------
# PROCESAR UN PDF CON UN MODELO
# ---------------------------------------------
def analizar_pdf_con_modelo(ruta_modelo, ruta_pdf, temperatura, prompt_base, coleccion):
    tokenizer, model = cargar_modelo_torch(ruta_modelo)

    print(f"📄 Procesando PDF: {ruta_pdf}")

    # Extraer texto del PDF
    try:
        doc = fitz.open(ruta_pdf)
    except:
        print(f"❌ Error al abrir el PDF: {ruta_pdf}")
        return None

    texto = ""
    for pagina in doc:
        texto += pagina.get_text("text") + "\n"
    doc.close()

    if not texto.strip():
        print("⚠ No se encontró texto en el PDF.")
        return None

    # Construir prompt final
    prompt = f"""{prompt_base}

Texto a analizar:
\"\"\"{texto[:4000]}\"\"\"

Asistente:
"""

    # Ejecución
    respuesta = generar_con_torch(tokenizer, model, prompt, temperatura)

    # Extraer JSON generado
    bloques_json = re.findall(r"\{[\s\S]*\}", respuesta)
    json_limpio = bloques_json[-1] if bloques_json else respuesta

    # Parsear JSON
    try:
        data = json.loads(json_limpio)
    except json.JSONDecodeError:
        print("⚠ JSON inválido. Guardando RAW.")
        data = {"respuesta_raw": respuesta}

    # Guardar archivo
    nombre_modelo = ruta_modelo.split("/")[-1]
    nombre_pdf = os.path.basename(ruta_pdf).replace(".pdf", "")

    nombre_salida = f"resumen_TORCH_{nombre_modelo}_{nombre_pdf}.json"

    with open(nombre_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 JSON guardado: {nombre_salida}")

    # Guardar en MongoDB
    coleccion.insert_one({
        "archivo_pdf": ruta_pdf,
        "modelo": ruta_modelo,
        "temperatura": temperatura,
        "resultado": data
    })

    print("📥 Guardado en MongoDB\n")

    return data


# ---------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    ruta_config = "F:/Git/TFG/config.txt"
    modelos, documentos, temperatura, prompt_base = leer_configuracion(ruta_config)

    print("\n=== MODELOS DETECTADOS ===")
    for m in modelos:
        print(" -", m)

    print("\n=== DOCUMENTOS DETECTADOS ===")
    for d in documentos:
        print(" -", d)

    print(f"\nTEMPERATURA: {temperatura}")
    print("\nPROMPT CARGADO (preview):")
    print(prompt_base[:200] + "...\n")

    # Conexión a MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["analisis_pdfs"]
    coleccion = db["resumenes_torch"]

    # Procesamiento global
    for modelo in modelos:
        for documento in documentos:
            analizar_pdf_con_modelo(modelo, documento, temperatura, prompt_base, coleccion)

    print("\n=== PROCESO COMPLETADO (TORCH) ===\n")
