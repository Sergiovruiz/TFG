from llama_cpp import Llama
import fitz  # PyMuPDF
import json
import re
import math

# === CONFIGURACIÓN DEL MODELO ===
llm = Llama(
    model_path=r"F:\\TFG\\text-generation-webui-3.13\\user_data\\models\\gpt-oss-20b-Q4_K_M.gguf",
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
print(f"\n📄 Procesando PDF: {ruta_pdf}\n")

# === 2. EXTRAER TEXTO COMPLETO ===
doc = fitz.open(ruta_pdf)
texto = "\n".join(pagina.get_text("text") for pagina in doc)
doc.close()

if not texto.strip():
    print("⚠️ No se ha encontrado texto en el PDF (puede ser escaneado).")
    exit()

# === 3. DIVIDIR TEXTO EN FRAGMENTOS (4000 caracteres por bloque) ===
bloques = [texto[i:i + 4000] for i in range(0, len(texto), 4000)]
total = len(bloques)
resumen_final = []

print(f"🧠 El documento se dividirá en {total} fragmentos para procesarlo por partes.\n")

# === 4. PROCESAR CADA BLOQUE CON EL MODELO ===
for i, bloque in enumerate(bloques, start=1):
    print(f"🔹 Analizando bloque {i}/{total}...")

    prompt = f"""
Eres un asistente que analiza documentos en español y devuelve la información exclusivamente en formato JSON.

Instrucciones importantes:
- No repitas, resumas ni menciones las instrucciones anteriores.
- No incluyas texto fuera del JSON.
- Resume solo el contenido del documento, no el prompt.
- Devuelve ÚNICAMENTE el JSON con esta estructura:

{{
  "resumen": [
    {{
      "titulo": "Título del apartado",
      "contenido": "Resumen breve del apartado."
    }},
    ...
  ]
}}

Texto a analizar (bloque {i}/{total}):
\"\"\"{bloque}\"\"\"
Asistente:
"""

    output = llm(
        prompt,
        max_tokens=700,
        temperature=0.6,
        top_p=0.9,
        stop=["Usuario:", "Asistente:"],
        echo=False
    )

    respuesta = output["choices"][0]["text"].strip()

    # Extraer y limpiar JSON
    bloques_json = re.findall(r"\{[\s\S]*\}", respuesta)
    if not bloques_json:
        continue

    json_limpio = bloques_json[-1]
    json_limpio = json_limpio.replace("<|end|>", "").replace("<|start|>", "").replace("<|channel|>", "").strip()

    try:
        data = json.loads(json_limpio)
        if isinstance(data, dict) and "resumen" in data:
            resumen_final.extend(data["resumen"])
    except json.JSONDecodeError:
        continue

# === 5. GUARDAR RESULTADO FINAL ===
resultado = {"resumen": resumen_final}

with open("resumen_completo.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

print("\n✅ Resumen completo generado y guardado en 'resumen_completo.json'\n")
print(json.dumps(resultado, indent=2, ensure_ascii=False))
