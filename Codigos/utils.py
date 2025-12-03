import json
import os

def construir_prompt(prompt_base, texto_pdf):
    return f"""{prompt_base}

Texto a analizar:
\"\"\"{texto_pdf[:4000]}\"\"\"
Asistente:
"""

def guardar_json(data, modelo, pdf, temperatura, prompt_id, carpeta_salida):
    nombre_modelo = os.path.basename(modelo).replace(".gguf", "")
    nombre_pdf = os.path.basename(pdf).replace(".pdf", "")

    nombre_archivo = f"resumen_{nombre_modelo}_{nombre_pdf}_T{temperatura}_P{prompt_id}.json"

    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

    with open(ruta_completa, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return ruta_completa
