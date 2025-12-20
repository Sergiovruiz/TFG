import os
import json
import csv

def construir_prompt(texto_pdf, modelo, pdf, temperatura, prompt_id):
    return f"""
You are a scientific reviewer specialized in experimental software engineering.

IMPORTANT INSTRUCTIONS (MUST FOLLOW):
- Respond ONLY with a single, valid JSON object.
- Do NOT include explanations, markdown, comments, or extra text.
- Do NOT repeat the JSON.
- Do NOT wrap the output in ```json blocks.

The JSON output MUST follow EXACTLY this structure:

{{
  "metadata": {{
    "modelo": "{modelo}",
    "documento": "{pdf}",
    "temperatura": {temperatura},
    "prompt_id": "{prompt_id}"
  }},
  "review": {{
    "Q1.1": {{
      "answer": "Yes|No|N/A",
      "justification": "..."
    }},
    "...": {{}},
    "Q10": {{
      "answer": "Yes|No|N/A",
      "justification": "..."
    }}
  }}
}}

Your task:
Evaluate the following scientific article using the checklist below.
For EACH question (Q1.1 to Q10):
- "answer" MUST be exactly one of: "Yes", "No", or "N/A"
- "justification" MUST be concise (maximum 2 sentences) and based ONLY on the article text.

Checklist:
Q1.1 Are null hypotheses explicitly defined?
Q1.2 Are alternative hypotheses explicitly defined?
Q2 Has the required sample size been calculated?
Q3 Have subjects been randomly selected?
Q4 Have subjects been randomly assigned to treatments?
Q5 Have test assumptions (normality, heteroskedasticity) been checked or discussed?
Q6 Has the definition of linear models been discussed?
Q7 Have the analysis results been interpreted using statistical concepts (p-values, confidence intervals, power)?
Q8 Do researchers avoid calculating and discussing post hoc power?
Q9 Is multiple testing (e.g., Bonferroni correction) reported?
Q10 Are descriptive statistics (means, counts) reported?

Article text to evaluate:

\"\"\"{texto_pdf[:3000]}\"\"\"

Respond ONLY with the JSON object described above.
"""

def guardar_json(data, modelo, pdf, temperatura, prompt_id, carpeta_salida):

    nombre_modelo = os.path.basename(modelo).replace(".gguf", "")
    nombre_pdf = os.path.basename(pdf).replace(".pdf", "")

    nombre_base = f"resumen_{nombre_modelo}_{nombre_pdf}_T{temperatura}_P{prompt_id}"
    extension = ".json"

    # Ruta inicial
    ruta = os.path.join(carpeta_salida, nombre_base + extension)

    # Si no existe, se guarda directamente
    if not os.path.exists(ruta):
        ruta_final = ruta
    else:
        # Buscar siguiente versión disponible
        version = 1
        while True:
            nombre_versionado = f"{nombre_base}_v{version}{extension}"
            ruta_versionada = os.path.join(carpeta_salida, nombre_versionado)

            if not os.path.exists(ruta_versionada):
                ruta_final = ruta_versionada
                break

            version += 1
    # Guardar archivo
    with open(ruta_final, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return ruta_final

def escribir_en_csv(ruta_csv, metadata, review):
    filas = []

    with open(ruta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            filas.append(fila)

    for fila in filas:
        if (fila["modelo"] == metadata["modelo"]
            and fila["documento"] == metadata["documento"]
            and float(fila["temperatura"]) == float(metadata["temperatura"])
            and int(fila["prompt_id"]) == int(metadata["prompt_id"])):
            pregunta = fila["pregunta"]

            if pregunta in review:
                fila["respuesta"] = review[pregunta]["answer"]
                fila["justificacion"] = review[pregunta]["justification"]

    # Reescribir CSV
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)