import csv
import numpy as np
import os

def crear_matriz(modelos, documentos, temperaturas, prompts):
    return np.array(np.meshgrid(modelos, documentos, temperaturas, prompts)).T.reshape(-1, 4)

def crear_csv(modelos, documentos, temperaturas, prompts, repeticiones, ruta_csv):

    combinaciones = crear_matriz(modelos, documentos, temperaturas, prompts)

    # Asegurar que existe la carpeta destino
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)

    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Cabecera FINAL
        writer.writerow(["modelo", "documento", "temperatura", "prompt_id", "repeticion", "pregunta", "respuesta", "justificacion"])

        # Lista fija de preguntas del checklist
        preguntas = ["Q1.1","Q1.2","Q2","Q3","Q4", "Q5","Q6","Q7","Q8","Q9","Q10"]

        # Filas: una por combinación × repetición × pregunta
        for modelo, documento, temperatura, prompt in combinaciones:
            prompt_id = prompts.index(prompt) + 1 if prompt != "DEFAULT" else "Default"

            for rep in range(1, repeticiones + 1):
                for pregunta in preguntas:
                    writer.writerow([modelo, documento, temperatura, prompt_id, rep, pregunta, "", ""])