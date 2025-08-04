from flask import Flask, render_template, request, jsonify
from main import get_bot_response  # We'll define this function below

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message')
    response = get_bot_response(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
