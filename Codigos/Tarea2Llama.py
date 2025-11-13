from llama_cpp import Llama
import fitz  # PyMuPDF
import json
import re

# === CONFIGURACIÓN DEL MODELO ===
llm = Llama(
    model_path=r"F:\\TFG\\text-generation-webui-3.13\\user_data\\models\\Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    n_ctx=2048,
    n_threads=12,
    n_gpu_layers=30,
    n_batch=512,
    use_mlock=True,
    use_mmap=True,
    verbose=False
)

# === 1. PEDIR RUTA DEL PDF ===
ruta_pdf = input("Introduce la ruta del archivo PDF: ").strip().strip('"')
print(f"\n Procesando PDF: {ruta_pdf}\n")

# === 2. EXTRAER TEXTO DEL PDF ===
doc = fitz.open(ruta_pdf)
texto = ""
for pagina in doc:
    texto += pagina.get_text("text") + "\n"
doc.close()

if not texto.strip():
    print(" No se ha encontrado texto en el PDF (puede ser escaneado).")
    exit()

# === 3. CREAR PROMPT DETALLADO ===
prompt = f"""
Eres un asistente que analiza documentos en español y devuelve la información exclusivamente en formato JSON.

Instrucciones importantes:
- No repitas, resumas ni menciones las instrucciones anteriores.
- No incluyas ningún texto fuera del JSON.
- No comentes lo que estás haciendo.
- Devuelve únicamente un JSON con el siguiente formato:

{{
  "resumen": [
    {{
      "titulo": "Título del apartado",
      "contenido": "Resumen breve del apartado."
    }},
    ...
  ]
}}

Texto a analizar:
\"\"\"{texto[:4000]}\"\"\"  # Limitamos el contexto
Asistente:
"""

# === 4. LLAMAR AL MODELO ===
output = llm(
    prompt,
    max_tokens=700,
    temperature=0.6,
    top_p=0.9,
    stop=["Usuario:", "Asistente:"],
    echo=False
)

respuesta = output["choices"][0]["text"].strip()

# === 5. EXTRAER BLOQUE JSON ===
bloques_json = re.findall(r"\{[\s\S]*\}", respuesta)
json_limpio = bloques_json[-1] if bloques_json else respuesta

# Limpieza de tokens especiales
json_limpio = (
    json_limpio.replace("<|end|>", "")
    .replace("<|start|>", "")
    .replace("<|channel|>", "")
    .replace("<|message|>", "")
    .strip()
)

# === 6. GUARDAR DIRECTAMENTE EN JSON LEGIBLE ===
try:
    data = json.loads(json_limpio)
    # Si dentro viene anidado como texto JSON (caso que mencionas)
    if isinstance(data, dict) and "respuesta_raw" in data:
        try:
            data = json.loads(data["respuesta_raw"])
        except json.JSONDecodeError:
            pass

    with open("resumen.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ JSON válido generado y guardado en 'resumen.json':\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))

except json.JSONDecodeError as e:
    print(f"⚠️ Error al parsear el JSON, se guardará la salida cruda.\n({e})")
    with open("resumen_raw.json", "w", encoding="utf-8") as f:
        json.dump({"respuesta_raw": respuesta}, f, indent=2, ensure_ascii=False)
    print("📁 Se ha guardado el contenido crudo en 'resumen_raw.json'.")
