from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Change for production

# Load the pipeline (which includes preprocessing and classifier)
try:
    model = joblib.load("xgboost_pipeline_model2.pkl")
except FileNotFoundError:
    print("Model file not found. Please ensure 'xgboost_pipeline_model2.pkl' exists.")
    exit()

# Features expected from the form (must match training features)
features = ['LVOT gradient', 'IVSd z-score', 'Age at first echo', 'LVPWd z-score']

@app.route("/prediction/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # 1. Get form data and validate
            input_data = [float(request.form[feature]) for feature in features]
            input_df = pd.DataFrame([input_data], columns=features)

            # 2. Predict using the pipeline (includes preprocessing)
            prob = model.predict_proba(input_df)[0][1]
            label = "High Risk" if prob >= 0.241 else "Low Risk"

            return render_template("results.html",
                                 risk_score=round(prob, 3),
                                 risk_label=label)
        except ValueError:
            flash("Invalid input: Please ensure all fields are filled with numbers.")
            return redirect(url_for('predict'))
        except Exception as e:
            flash(f"An unexpected error occurred: {e}")
            return redirect(url_for('predict'))
    # For GET request, render the form
    return render_template("modelForm.html")

@app.route("/")
def home():
    return redirect(url_for('predict'))

if __name__ == "__main__":
    app.run(debug=True)
