
import pandas as pd
import joblib

model = joblib.load("movie_genre.pk1")

print("\n========================================")
print("       MOVIE GENRE CLASSIFICATION")
print("========================================")

print("\n<------ Enter Movie Details ------>\n")

description = input("Enter Movie Description: ")
production_company = input("Enter Production Company: ")
country = input("Enter the country name: ")

movie_data = {
    "Description": description,
    "Production_Company": production_company ,
    "Country": country
}

new_movie = pd.DataFrame([movie_data])

prediction = model.predict(new_movie)[0]
probabilities = model.predict_proba(new_movie)[0]
probability = probabilities[list(model.classes_).index(prediction)]

print("\n========================================")
print("          PREDICTION RESULT")
print("========================================")

print("Predicted Genre:", prediction)
print(f"Confidence: {probability * 100:.2f}%")