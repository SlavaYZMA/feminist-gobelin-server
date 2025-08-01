import os
import json
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": os.environ.get("CORS_ORIGINS", "*")}})

try:
    with open("resources.json", "r", encoding="utf-8") as f:
        resources = json.load(f)
except FileNotFoundError:
    resources = []

try:
    with open("users.json", "r", encoding="utf-8") as f:
        users = json.load(f)
except FileNotFoundError:
    users = []

print("Загружаю модель...")
model = AutoModelForCausalLM.from_pretrained(
    "SlavaYZMA/feminist-gobelin-model",
    use_safetensors=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    token=os.environ.get("HF_TOKEN")
)
tokenizer = AutoTokenizer.from_pretrained("SlavaYZMA/feminist-gobelin-model", token=os.environ.get("HF_TOKEN"))
print("Модель загружена!")

def find_resource(prompt, country, problem):
    for resource in resources:
        if (resource["country"].lower() in country.lower() and 
            resource["type"].lower() in problem.lower()):
            return f"В {resource['country']}: Звоните {resource['helpline']}. Закон: {resource['law']}. Действия: {resource['action']}"
    return None

def save_user(user_id, name, country, problem):
    for user in users:
        if user["user_id"] == user_id:
            user.update({"name": name, "country": country, "problem": problem})
            break
    else:
        users.append({"user_id": user_id, "name": name, "country": country, "problem": problem})
    with open("users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_id = data.get('user_id', '')
    name = data.get('name', '')
    country = data.get('country', '')
    problem = data.get('problem', '')
    prompt = data.get('prompt', '')

    if user_id and (name or country or problem):
        save_user(user_id, name, country, problem)

    full_prompt = f"{name}, в {country}, проблема: {problem}. {prompt}" if name and country and problem else prompt

    json_response = find_resource(full_prompt, country, problem)
    if json_response:
        return jsonify({'response': json_response})

    with torch.no_grad():
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.5,
            top_p=0.95,
            do_sample=False,
            num_beams=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': response + " (Проверьте информацию у официальных источников.)"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)