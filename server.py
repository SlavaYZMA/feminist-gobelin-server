from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех запросов

# Загружаем модель и токенизатор
try:
    model = AutoModelForCausalLM.from_pretrained("./my_feminist_model", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained("./my_feminist_model")
    tokenizer.pad_token = tokenizer.eos_token
    print("Модель загружена успешно!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.json
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'Пустой запрос'}), 400
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)