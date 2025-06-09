from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan encoder
model = joblib.load("./models/rf_model.pkl")
label_encoders = joblib.load("./models/label_encoders.pkl")
target_encoder = joblib.load("./models/target_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Ambil data dari form
            input_data = {
                "age": int(request.form["age"]),
                "gender": request.form["gender"],
                "employment_status": request.form["employment_status"],
                "work_environment": request.form["work_environment"],
                "mental_health_history": request.form["mental_health_history"],
                "seeks_treatment": request.form["seeks_treatment"],
                "stress_level": int(request.form["stress_level"]),
                "sleep_hours": float(request.form["sleep_hours"]),
                "physical_activity_days": int(request.form["physical_activity_days"]),
                "depression_score": int(request.form["depression_score"]),
                "anxiety_score": int(request.form["anxiety_score"]),
                "social_support_score": int(request.form["social_support_score"]),
                "productivity_score": float(request.form["productivity_score"]),
            }

            # Encode data
            for col, le in label_encoders.items():
                input_data[col] = le.transform([input_data[col]])[0]

            # Konversi ke array
            features = np.array(list(input_data.values())).reshape(1, -1)

            # Prediksi
            pred_encoded = model.predict(features)[0]
            prediction = target_encoder.inverse_transform([pred_encoded])[0]

        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
