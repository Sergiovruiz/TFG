import csv
import numpy as np
import os

def crear_matriz(modelos, documentos, temperaturas, prompts):
    return np.array(np.meshgrid(modelos, documentos, temperaturas, prompts)).T.reshape(-1, 4)

def crear_csv(modelos,documentos,temperaturas,prompts,repeticiones,ruta_csv):

    combinaciones = crear_matriz(modelos, documentos, temperaturas, prompts)

    # Asegurar que existe la carpeta destino
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)

    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Cabecera
        writer.writerow(["modelo","documento","temperatura","prompt_id","repeticion","resultado"])

        # Filas
        for modelo, documento, temperatura, prompt in combinaciones:
            prompt_id = prompts.index(prompt) + 1

            for rep in range(1, repeticiones + 1):
                writer.writerow([modelo,documento,temperatura,prompt_id,rep,""])