from llama_cpp import Llama
import fitz  # PyMuPDF
import json
import re
from pymongo import MongoClient

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

# === CONEXIÓN A MONGODB ===
client = MongoClient("mongodb://localhost:27017/")  # Cambia si usas otro host/puerto
db = client["analisis_pdfs"]
coleccion = db["resumenes"]

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
# === 3. CREAR PROMPT DETALLADO ===
prompt = f"""
Eres un asistente que analiza documentos en español y devuelve exclusivamente un JSON con una estructura fija, para que todos los resultados sean comparables entre documentos.

Instrucciones importantes:
- Devuelve SOLO el JSON, sin texto adicional.
- Respeta exactamente la estructura mostrada.
- Si algún apartado no existe en el texto, deja su contenido vacío ("") pero mantenlo en el JSON.
- Mantén siempre el mismo orden y los mismos nombres de campos.
- No traduzcas ni modifiques los nombres de los apartados.
- No añadas ni elimines secciones.

Estructura fija del JSON:

{{
  "documento": {{
    "titulo": "Título general del documento (si se puede identificar)",
    "autor": "Autor o entidad responsable del documento (si aparece)",
    "fecha": "Fecha de creación o publicación (si aparece)",
    "resumen_general": "Resumen breve del contenido total del documento.",
    "apartados": [
      {{
        "titulo": "Introducción",
        "contenido": "Resumen del contexto o propósito del documento.",
        "palabras_clave": []
      }},
      {{
        "titulo": "Objetivos",
        "contenido": "Descripción de los objetivos principales del documento.",
        "palabras_clave": []
      }},
      {{
        "titulo": "Metodología",
        "contenido": "Resumen de los métodos, procesos o enfoques utilizados.",
        "palabras_clave": []
      }},
      {{
        "titulo": "Resultados",
        "contenido": "Resumen de los hallazgos, análisis o conclusiones parciales.",
        "palabras_clave": []
      }},
      {{
        "titulo": "Conclusiones",
        "contenido": "Síntesis final y posibles recomendaciones.",
        "palabras_clave": []
      }}
    ]
  }}
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

# === 6. GUARDAR EN JSON Y EN MONGODB ===
try:
    data = json.loads(json_limpio)
    if isinstance(data, dict) and "respuesta_raw" in data:
        try:
            data = json.loads(data["respuesta_raw"])
        except json.JSONDecodeError:
            pass

    # Guarda en archivo
    with open("resumen.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Guarda en MongoDB
    documento = {
        "archivo": ruta_pdf,
        "contenido": data
    }
    coleccion.insert_one(documento)

    print("\nJSON válido guardado en 'resumen.json' y en MongoDB.\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))

except json.JSONDecodeError as e:
    print(f"Error al parsear el JSON, se guardará la salida.\n({e})")
    with open("resumen_raw.json", "w", encoding="utf-8") as f:
        json.dump({"respuesta_raw": respuesta}, f, indent=2, ensure_ascii=False)

    # Guarda también la versión cruda en MongoDB
    coleccion.insert_one({
        "archivo": ruta_pdf,
        "error": str(e),
        "respuesta_raw": respuesta
    })
    print("Se ha guardado el contenido en 'resumen_raw.json' y en MongoDB.")
