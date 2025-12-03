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

def extraer_json(texto):
    bloques = re.findall(r"\{[\s\S]*\}", texto)
    contenido = bloques[-1] if bloques else texto
    contenido = contenido.replace("<|end|>", "").strip()

    try:
        return json.loads(contenido)
    except:
        return {"respuesta_raw": texto}
