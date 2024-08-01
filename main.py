from flask import Flask, request, jsonify
from groq import Groq
import os

app = Flask(__name__)

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/ai', methods=['POST'])
def ai_endpoint():
    data = request.json
    model = data.get('model', 'mixtral-8x7b-32768')
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return jsonify({"response": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))