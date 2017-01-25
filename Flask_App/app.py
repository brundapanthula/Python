import os
from flask import Flask,render_template,request,jsonify,json
from model_build import house_price
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload",methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT,'input/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
    return render_template("complete.html")

@app.route("/learning",methods=['POST'])
def learning():
    data = ".\input"
    # data == {"userInput": "whatever text you entered"}
    response = house_price(data)
    return jsonify(response)


if __name__ == '__main__':
    app.run(port=4555,debug = True)


