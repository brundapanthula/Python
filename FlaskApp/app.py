from flask import Flask
import os
from flask import render_template, jsonify, request,redirect, json
from model_build import house_price
app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learning', methods=['POST'])
def learning():
    data = json.loads(request.data)
    # data == {"userInput": "whatever text you entered"}
    response = house_price(data)
    return response

if __name__ == '__main__':
    app.run(debug = True)
