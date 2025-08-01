import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": os.environ.get("CORS_ORIGINS", "*")}})

print("Загружаю модель...")
model = AutoModelForCausalLM.from_pretrained("SlavaYZMA/feminist-gobelin-model", use_safetensors=True, token=os.environ.get("HF_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("SlavaYZMA/feminist-gobelin-model", token=os.environ.get("HF_TOKEN"))
print("Модель загружена успешно!")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)