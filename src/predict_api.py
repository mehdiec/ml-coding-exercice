from flask import Flask, render_template, request
from flask_restful import Api
import pickle

from predict import predict_one_shot

DIM = 28, 28
app = Flask(__name__)
api = Api(app)

model = pickle.load(open("./models/svm_1/model.pck", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def predict_test():
    prediction = 0
    img_path = "./data/6.jpg"
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = "./data/" + img.filename
        img.save(img_path)
        prediction = prediction = predict_one_shot(model, img_path)

    return render_template("submit.html", prediction=prediction, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True, port="1080")
