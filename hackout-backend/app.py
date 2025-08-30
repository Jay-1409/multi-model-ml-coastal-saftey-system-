from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load hurricane model
with open("models_pkl/wave_height_model.pkl", "rb") as f:
    hurricane_model = pickle.load(f)

# Load tsunami model
with open("models_pkl/tsunami_model.pkl", "rb") as f:
    tsunami_model = pickle.load(f)

# Hurricane prediction route
@app.route("/hurricane", methods=["POST"])
def predict_hurricane():
    data = request.json
    
    # Extract all 14 features in the same order as training
    features = np.array([[
        data["wind_speed"], 
        data["pressure"], 
        data["latitude"], 
        data["longitude"], 
        data["storm_category"],
        data["humidity"],
        data["wind_direction"],
        data["wave_height"],
        data["month"],
        data["day_of_year"],
        data["sea_surface_temp"],
        data["cloud_cover"],
        data["precipitation"],
        data["storm_radius"]
    ]])
    
    # Predict probability if supported
    if hasattr(hurricane_model, "predict_proba"):
        prob = hurricane_model.predict_proba(features)[0][1]
        return jsonify({"rough_sea_probability": prob})
    
    # Otherwise, predict class
    pred = hurricane_model.predict(features)[0]
    return jsonify({"rough_sea": bool(pred)})

# Tsunami prediction route
@app.route("/tsunami", methods=["POST"])
def predict_tsunami():
    data = request.json

    # Mapping for categorical feature
    magType_mapping = {
        "mb": 0,
        "ms": 1,
        "mw": 2,
        "ml": 3,
        "mwr": 4
    }

    # Columns must match training
    feature_columns = [
        "magnitude", "cdi", "mmi", "sig", "nst", "dmin", "gap", "depth",
        "latitude", "longitude", "Year", "Month", "magType"
    ]

    # Create a DataFrame so feature names match
    features_df = pd.DataFrame([[
        data["magnitude"],
        data["cdi"],
        data["mmi"],
        data["sig"],
        data["nst"],
        data["dmin"],
        data["gap"],
        data["depth"],
        data["latitude"],
        data["longitude"],
        data["Year"],
        data["Month"],
        magType_mapping[data["magType"]]
    ]], columns=feature_columns)

    # Predict class
    prediction = tsunami_model.predict(features_df)

    # Predict probability if available
    if hasattr(tsunami_model, "predict_proba"):
        probability = tsunami_model.predict_proba(features_df)[0][1]
        return jsonify({
            "tsunami_prediction": int(prediction[0]),
            "probability": float(probability)
        })

    return jsonify({"tsunami_prediction": int(prediction[0])})


# Load algae model
with open("models_pkl/algae_bloom.pkl", "rb") as f:
    algae_model = pickle.load(f)


#This indicates the predicted algae concentration in µg/L.
@app.route("/algae", methods=["POST"])
def predict_algae():
    data = request.json

    # Features exactly as used during model training
    features = np.array([[
        data["Hs"],
        data["Hmax"],
        data["Tz"],
        data["Tp"],
        data["SST"]
    ]])

    # Predict probability if supported
    if hasattr(algae_model, "predict_proba"):
        prob = algae_model.predict_proba(features)[0][1]
        return jsonify({"algae_probability": float(prob)})

    # Otherwise, predict value (regressor)
    pred = algae_model.predict(features)[0]
    return jsonify({"algae_prediction": float(pred)})


with open("models_pkl/bleaching.pkl", "rb") as f:
    logistic_model = pickle.load(f)

@app.route("/bleaching", methods=["POST"])
def predict():
    try:
        # Expect JSON input: {"features": [val1, val2, val3, ...]}
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = logistic_model.predict(features)[0]
        probability = logistic_model.predict_proba(features)[0].tolist()

        return jsonify({
            "prediction": int(prediction),   # class label
            "probability": probability       # probabilities for each class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_all", methods=["POST"])
def predict_all():
    data = request.get_json()
    results = {}
    print(data)
    def safe_get(d, key, default=0):
        """Safely get value from dict, return default if missing or None."""
        return d.get(key, default) if d and d.get(key) is not None else default

    # 1. Hurricane
    try:
        hurricane = data.get("hurricane", {})
        hurricane_features = np.array([[
            safe_get(hurricane, "wind_speed", 0),
            safe_get(hurricane, "pressure", 1013),
            safe_get(hurricane, "latitude", 0),
            safe_get(hurricane, "longitude", 0),
            safe_get(hurricane, "storm_category", 1),
            safe_get(hurricane, "humidity", 50),
            safe_get(hurricane, "wind_direction", 0),
            safe_get(hurricane, "wave_height", 0),
            safe_get(hurricane, "month", 1),
            safe_get(hurricane, "day_of_year", 1),
            safe_get(hurricane, "sea_surface_temp", 25),
            safe_get(hurricane, "cloud_cover", 110),
            safe_get(hurricane, "precipitation", 0),
            safe_get(hurricane, "storm_radius", 50)
        ]])

        if hasattr(hurricane_model, "predict_proba"):
            results["hurricane"] = float(hurricane_model.predict_proba(hurricane_features)[0][1])
        else:
            results["hurricane"] = int(hurricane_model.predict(hurricane_features)[0])
    except Exception as e:
        results["hurricane_error"] = str(e)

    # 2. Tsunami
    try:
        tsunami = data.get("tsunami", {})
        magType_mapping = {"mb": 0, "ms": 1, "mw": 2, "ml": 3, "mwr": 4}
        tsunami_features = pd.DataFrame([[
            safe_get(tsunami, "magnitude", 5.0),
            safe_get(tsunami, "cdi", 1),
            safe_get(tsunami, "mmi", 1),
            safe_get(tsunami, "sig", 100),
            safe_get(tsunami, "nst", 10),
            safe_get(tsunami, "dmin", 0.1),
            safe_get(tsunami, "gap", 50),
            safe_get(tsunami, "depth", 10),
            safe_get(tsunami, "latitude", 0),
            safe_get(tsunami, "longitude", 0),
            safe_get(tsunami, "Year", 2025),
            safe_get(tsunami, "Month", 1),
            magType_mapping.get(tsunami.get("magType", "mw"), 2)
        ]], columns=[
            "magnitude", "cdi", "mmi", "sig", "nst", "dmin", "gap", "depth",
            "latitude", "longitude", "Year", "Month", "magType"
        ])

        if hasattr(tsunami_model, "predict_proba"):
            results["tsunami"] = float(tsunami_model.predict_proba(tsunami_features)[0][1])
        else:
            results["tsunami"] = int(tsunami_model.predict(tsunami_features)[0])
    except Exception as e:
        results["tsunami_error"] = str(e)

    # 3. Algae
    try:
        algae = data.get("algae", {})
        algae_features = np.array([[
            safe_get(algae, "Hs", 1),
            safe_get(algae, "Hmax", 2),
            safe_get(algae, "Tz", 5),
            safe_get(algae, "Tp", 8),
            safe_get(algae, "SST", 26)
        ]])

        if hasattr(algae_model, "predict_proba"):
            results["algae"] = float(algae_model.predict_proba(algae_features)[0][1])
        else:
            results["algae"] = float(algae_model.predict(algae_features)[0])
    except Exception as e:
        results["algae_error"] = str(e)

    # 4. Coral Bleaching
    try:
        bleaching = data.get("bleaching", {})
        bleaching_features = np.array(bleaching.get("features", [25, 30, 5])).reshape(1, -1)

        prediction = logistic_model.predict(bleaching_features)[0]
        probability = logistic_model.predict_proba(bleaching_features)[0][1]

        results["bleaching"] = {
            "prediction": int(prediction),
            "probability": float(probability)
        }
    except Exception as e:
        results["bleaching_error"] = str(e)
    print(results)
    return jsonify(results)





if __name__ == "__main__":
    app.run(debug=True)



