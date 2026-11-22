import pandas as pd 
df = pd.read_csv("daraset/final_dataset.csv")
print(df.columns.to_list())
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
    "Director",
    "Production_Company",
    "Country"
])

y = df["Genre"]

print(X.columns.to_list())

numerical_feature = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_feature = X.select_dtypes(
    include=["object"]
).columns.tolist()

categorical_feature.remove("Description")


print("Numerical:", numerical_feature)
print("Categorical:", categorical_feature)
print("Text:", text_feature)