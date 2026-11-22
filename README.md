# 🎬 Movie Genre Classification

A Machine Learning project that predicts the genre of a movie using its textual description and available metadata.

The project uses **TF-IDF Vectorization, Scikit-learn Pipelines, and LightGBM** to process movie information and classify it into one of seven genres.

## 🚀 Project Overview

This project demonstrates a machine learning workflow for multi-class text classification.

Given movie information, the trained model predicts one of the following genres:

- Action
- Comedy
- Drama
- Fantasy
- Horror
- Romance
- Thriller

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- TF-IDF Vectorizer
- Joblib

## ⚙️ Machine Learning Workflow

1. Load and explore the movie dataset.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Handle missing numerical and categorical values.
5. Scale numerical features using MinMaxScaler.
6. Encode categorical features using OneHotEncoder.
7. Convert movie descriptions into numerical features using TF-IDF.
8. Combine preprocessing and classification using Scikit-learn Pipeline.
9. Train the LightGBM classification model.
10. Evaluate the model and save it using Joblib.

## 🧠 Model Details

**Algorithm:** LightGBM Classifier

**Text Processing:** TF-IDF Vectorization

**Numerical Preprocessing:** SimpleImputer and MinMaxScaler

**Categorical Preprocessing:** SimpleImputer and OneHotEncoder

**Model Parameters:**
- Number of estimators: 100
- Learning rate: 0.05
- Maximum depth: 5
- Random state: 42

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The current training run reported 100% training and testing accuracy. This result should be independently validated for data leakage and duplicate records before being treated as evidence of real-world performance.

## 💻 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Movie-Genre-Classification.git
```

### 2. Open the project folder

```bash
cd Movie-Genre-Classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```


### 6. Train the model

```bash
python train.py
```

### 7. Predict a movie genre

```bash
python predict.py
```

Enter the movie details when prompted to receive a predicted genre and its model confidence score.

## 📁 Project Files

| File | Description |
|---|---|
| `train.py` | Trains and saves the classification model |
| `predict.py` | Predicts the genre of a new movie |
| `movie_genre.pk1` | Saved machine learning pipeline |
| `dataset/` | Movie classification dataset, if included |

## 🔮 Future Improvements

- Improve prediction performance on unseen movie descriptions.
- Validate the model using cross-validation.
- Experiment with different classification algorithms.
- Add a web interface using Flask.
- Build a more user-friendly movie prediction application.

## 👨‍💻 Author

Developed as a Machine Learning project to explore text classification, feature engineering, preprocessing pipelines, and model deployment using Python.

---

⭐ If you find this project interesting, consider giving the repository a star!
