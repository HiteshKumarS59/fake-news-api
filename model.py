import pandas as pd
import re
import pickle
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Label the data
df_fake["label"] = 0
df_real["label"] = 1

# Combine and shuffle
df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42).reset_index(drop=True)
df["text"] = df["title"] + " " + df["text"]

# Clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().split()
    return ' '.join([word for word in text if word not in stop_words and len(word) > 1])

df["text"] = df["text"].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Evaluate
metrics = {}
best_accuracy = 0
best_model = None

print("🔍 Evaluating models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n📊 {name} Classification Report:\n")
    print(classification_report(y_test, preds))

    metrics[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'model': model,
        'conf_matrix': confusion_matrix(y_test, preds)
    }

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# Save best model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n✅ Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
print("🧠 Model and Vectorizer saved successfully!")

# --- Visualization ---

# Plot metrics
df_metrics = pd.DataFrame(metrics).T[['accuracy', 'precision', 'recall', 'f1']]
df_metrics.plot(kind='bar', figsize=(10,6), colormap='coolwarm')
plt.title("📊 Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Plot confusion matrix for best model
sns.heatmap(metrics[best_model_name]['conf_matrix'], annot=True, fmt='d', cmap='Greens')
plt.title(f"🔍 Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
