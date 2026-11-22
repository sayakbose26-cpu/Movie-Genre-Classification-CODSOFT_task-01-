import pandas as pd 
df = pd.read_csv("daraset/movie_genre_classification_final.csv")
print(df.columns.to_list())
X = df.drop(columns=["Genre",'Title',"Duration",'Rating', 'Votes','Budget_USD', 'BoxOffice_USD', 'Year', 'Content_Rating', 'Lead_Actor', 'Num_Awards', 'Critic_Reviews',"Year","Director"])
y = df["Genre"]
print(X.columns.to_list())
print(y.name)
numerical_feature = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_feature = X.select_dtypes(
    include=["object"]
).columns.tolist()

text_feature = ["Description"]

print("Numerical:", numerical_feature)
print("Categorical:", categorical_feature)
print("Text:", text_feature)
print(df[["Description", "Genre"]].sample(10, random_state=42))