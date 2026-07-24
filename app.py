from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# Load Model and Pipeline
# --------------------------------------------------
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    longitude = float(request.form["longitude"])
    latitude = float(request.form["latitude"])
    housing_median_age = int(request.form["housing_median_age"])
    total_rooms = int(request.form["total_rooms"])
    total_bedrooms = int(request.form["total_bedrooms"])
    population = int(request.form["population"])
    households = int(request.form["households"])
    median_income = float(request.form["median_income"])
    ocean_proximity = request.form["ocean_proximity"]

    input_df = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

#I AM CHANGING
    print("\n===== INPUT DATA =====")
    print(input_df)
    transformed = pipeline.transform(input_df)
    prediction = model.predict(transformed)


    print("\n===== PREDICTION =====")
    print(prediction)

    price = round(float(prediction[0]), 2)

    chart_data = {
        "labels": ["Rooms", "Bedrooms", "Population", "Households"],
        "values": [
            total_rooms,
            total_bedrooms,
            population,
            households
        ]
    }

    return render_template(
        "result.html",
        prediction=price,
        chart_data=chart_data,
        latitude=latitude,
        longitude=longitude
    )


if __name__ == "__main__":
    app.run(debug=True)
