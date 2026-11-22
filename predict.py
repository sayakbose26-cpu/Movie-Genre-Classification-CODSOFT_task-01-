
import pandas as pd
import joblib

model = joblib.load("movie_genre.pk1")

df = pd.read_csv("daraset/movie_genre_classification_final.csv")

print("\n========================================")
print("       MOVIE GENRE CLASSIFICATION")
print("========================================")

print("\n<------ Enter Movie Details ------>\n")

description = input("Enter Movie Description: ")

movie_data = {
    "Description": description,
}

new_movie = pd.DataFrame([movie_data])

prediction = model.predict(new_movie)[0]

probabilities = model.predict_proba(new_movie)[0]

probability = probabilities[
    list(model.classes_).index(prediction)
]

print("\n========================================")
print("          PREDICTION RESULT")
print("========================================")

print("Predicted Genre:", prediction)
print(f"Confidence: {probability * 100:.2f}%")

movies = df[df["Genre"] == prediction]

print("\n========================================")
print("   SOME RECOMMEND MOVIES OF", prediction)
print("========================================")

if not movies.empty:
    print(movies["Title"].dropna().drop_duplicates().head(6).to_string(index=False))
else:
    print("No movies found for this genre.")

print("\n========================================")
