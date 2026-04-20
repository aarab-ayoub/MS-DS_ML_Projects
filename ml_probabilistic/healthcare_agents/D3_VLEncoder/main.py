from pathlib import Path

from agents.encoder import VLEncoder, load_model


project_root = Path(__file__).resolve().parent
model_path = project_root / "models" / "qwen2.5-vl-3b-instruct-q4_k_m.gguf"
image_path = project_root / "data" / "path_vqa_000.jpg"

llm = load_model(str(model_path))
encoder = VLEncoder(llm)
encoder.warmup()

question = "what is this ?"
result = encoder.encode(str(image_path), question)

print("-------Question-----")
print(question)
print("-------Answer-----")
print(result)
metrics = encoder.get_last_benchmark()
print("-------metrics-----")
print(metrics)