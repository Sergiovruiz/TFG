import os
import httpx

HF_TOKEN = os.getenv("HF_TOKEN")  # tu token de Hugging Face
BASE_URL = "https://router.huggingface.co/v1"
MODEL = "openai/gpt-oss-20b:cerebras"

def ask_model(prompt: str) -> str:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    if not HF_TOKEN:
        print("⚠️ Define primero la variable de entorno HF_TOKEN con tu token de Hugging Face")
        exit(1)
    print("=== Chat CLI con gpt-oss-20b ===")
    while True:
        try:
            query = input("\nTu pregunta > ")
            if query.lower() in {"salir", "exit", "quit"}:
                print("Adiós 👋")
                break
            answer = ask_model(query)
            print(f"\n🤖 Respuesta:\n{answer}")
        except KeyboardInterrupt:
            print("\nAdiós 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

# import os

# from openai import OpenAI

# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.getenv("HF_TOKEN"),
# )

# while True:
#     query = input("\nTu pregunta > ")
#     if query.lower() in {"salir", "exit", "quit"}:
#         print("Adiós 👋")
#         break

#     completion = client.chat.completions.create(
#         model="openai/gpt-oss-20b:cerebras",  # ajusta proveedor si hace falta
#         messages=[
#             {"role": "user", "content": query}
#         ],
#         temperature=0.7,
#         max_tokens=300,
#     )

#     print("\n🤖 Respuesta:")
#     print(completion.choices[0].message.content)