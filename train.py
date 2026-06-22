# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load data
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# Label data
real["label"] = 1
fake["label"] = 0

# Combine
df = pd.concat([real, fake])
df = df.sample(frac=1).reset_index(drop=True)

# Features
X = df["title"] + " " + df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, pred)*100:.2f}%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model saved!")
