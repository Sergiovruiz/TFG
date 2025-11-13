from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

MODEL_PATH = Path("F:/TFG/Models/gpt-oss-20b").resolve()

config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_hidden_layers = config.num_hidden_layers // 2  # usar la mitad de capas

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    offload_folder="F:/TFG/offload"
)

print("✅ Modelo cargado parcialmente en GPU + RAM + disco.\n")

while True:
    query = input("🟢 Pregunta ('salir' para terminar): ")
    if query.lower() in ["salir", "exit", "quit"]:
        break

    inputs = tokenizer(query, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\n💬 Respuesta:\n")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("\n" + "-" * 50 + "\n")
