from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]

        features_array = np.array([features])

        # If using Random Forest (most likely)
        prediction = model.predict(features_array)

        result = label_encoder.inverse_transform(prediction)

        return render_template("result.html", prediction=result[0])

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)