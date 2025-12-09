from llama_cpp import Llama
import json
import re

def cargar_modelo(ruta_modelo):
    return Llama(
        model_path=ruta_modelo,
        n_ctx=2048,
        n_threads=12,
        n_gpu_layers=30,
        n_batch=512,
        use_mlock=True,
        use_mmap=True,
        verbose=False
    )

def ejecutar_modelo(llm, prompt, temperatura):
    resp = llm(
        prompt,
        max_tokens=700,
        temperature=temperatura,
        top_p=0.9,
        stop=["Usuario:", "Asistente:"],
        echo=False
    )
    return resp["choices"][0]["text"].strip()

def extraer_json(texto: str):
    texto = texto.strip()

    # Busca si es un JSON limpio
    try:
        return json.loads(texto)
    except Exception:
        pass

    # Busca un objeto que empiece por {"metadata"
    i = texto.find('{"metadata"')
    if i == -1:
        # Si no lo encuentra, buscar el primer '{'
        i = texto.find("{")

    if i != -1:
        bloque = _extraer_bloque_json(texto, i)
        if bloque:
            try:
                return json.loads(bloque)
            except Exception:
                pass

    # Si todo falla, devolvemos la salida cruda
    return {"respuesta_raw": texto}

def _extraer_bloque_json(texto, inicio):
    
    nivel = 0
    en_cadena = False
    escapar = False

    for i in range(inicio, len(texto)):
        ch = texto[i]

        if en_cadena:
            if escapar:
                escapar = False
            elif ch == "\\":
                escapar = True
            elif ch == '"':
                en_cadena = False
        else:
            if ch == '"':
                en_cadena = True
            elif ch == "{":
                nivel += 1
            elif ch == "}":
                nivel -= 1
                if nivel == 0:
                    return texto[inicio:i+1]

    return None

