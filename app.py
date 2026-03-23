from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model and preprocessing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Gather the raw inputs from the HTML form
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]

        # 2. Convert to a 2D NumPy array (required by scikit-learn)
        features_array = np.array([features])

        # 3. CRITICAL FIX: Scale the input data exactly how the training data was scaled
        scaled_features = scaler.transform(features_array)

        # 4. Predict using the scaled data
        prediction = model.predict(scaled_features)

        # 5. Decode the numerical prediction back into the readable crop name
        result = label_encoder.inverse_transform(prediction)

        # 6. Return the result AND the original inputs so result.html can display them
        return render_template("result.html", prediction=result[0], inputs=request.form)

    except Exception as e:
        # It's good practice to return the error for debugging, but in a real app, 
        # you'd want to return a friendly error page.
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Remove debug=True before final deployment, but keep it for local testing
    app.run(debug=True)