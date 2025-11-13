from llama_cpp import Llama

# Cargar el modelo GGUF
llm = Llama(
    model_path=r"F:\TFG\text-generation-webui-3.13\user_data\models\gpt-oss-20b-Q4_K_M.gguf",
    n_ctx=2048,          # 2048 tokens de contexto (puedes subirlo si tu RAM lo permite)
    n_threads=12,         # tu R5 5600 tiene 6 núcleos / 12 hilos → usa 12
    n_gpu_layers=30,      # usa GPU para ~30 capas (2060 de 12 GB lo aguanta bien)
    n_batch=512,          # mejora rendimiento sin saturar VRAM
    use_mlock=True,       # bloquea la RAM del modelo (evita swaps)
    use_mmap=True,        # carga mapeado a memoria (más eficiente)
    verbose=False         # limpia la consola
)

# Loop interactivo
print("Modelo cargado. Escribe 'salir' para terminar.\n")

SYSTEM_PROMPT = (
    "You are a helpful and concise assistant that answers questions clearly in Spanish.\n"
)

while True:
    query = input(">> ")
    if query.lower() in ["salir", "exit", "quit"]:
        break

    prompt = f"{SYSTEM_PROMPT}Usuario: {query}\nAsistente:"
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        stop=["Usuario:", "Asistente:"],
        echo=False
    )

    print(":", output["choices"][0]["text"].strip(), "\n")