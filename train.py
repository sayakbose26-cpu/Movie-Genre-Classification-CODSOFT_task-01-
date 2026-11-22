
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import joblib

df = pd.read_csv("daraset/movie_genre_classification_final.csv")

df["Description"] = df["Description"].fillna("").astype(str)

X = df.drop(columns=[
    "Genre",
    "Title",
    "Duration",
    "Rating",
    "Votes",
    "Budget_USD",
    "BoxOffice_USD",
    "Language",
    "Content_Rating",
    "Lead_Actor",
    "Num_Awards",
    "Critic_Reviews",
    "Year", 
    "Director"
    
])

y = df["Genre"]
print(X.columns.to_list())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numerical_feature = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_feature = X.select_dtypes(
    include=["object"]
).columns.tolist()

categorical_feature.remove("Description")

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

text_pipeline = TfidfVectorizer()

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_feature),
    ("categorical", categorical_pipeline, categorical_feature),
    ("text", text_pipeline, "Description")
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])

model.fit(X_train, y_train)


joblib.dump(model, "movie_genre.pk1")

print("Model saved successfully")
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

