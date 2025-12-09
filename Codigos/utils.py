import json
import os

def construir_prompt(prompt_base, texto_pdf, modelo, pdf, temperatura, prompt_id):
    return f"""
INSTRUCCIONES IMPORTANTES — LÉELAS CON ATENCIÓN:

Debes generar *únicamente un JSON válido*, sin texto añadido, sin explicaciones, sin formato Markdown,
y sin repetir el contenido varias veces.

El JSON final debe tener esta estructura EXACTA:

{{
  "metadata": {{
    "modelo": "{modelo}",
    "documento": "{pdf}",
    "temperatura": {temperatura},
    "prompt_id": {prompt_id}
  }},
  "analisis": {{
      // Aquí debes generar el contenido solicitado por el prompt base
  }}
}}

REGLAS ESTRICTAS:
- No escribas bloques ```json ni etiquetas Markdown.
- No generes más de un JSON. Solo uno.
- No repitas el JSON. No incluyas versiones múltiples.
- No incluyas comentarios dentro del JSON final (usa solo texto).
- No generes texto fuera del JSON.
- Sustituye el contenido de "analisis" por el resultado del prompt que viene a continuación.

--- PROMPT DEL USUARIO ---
{prompt_base}

--- TEXTO A ANALIZAR ---
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
