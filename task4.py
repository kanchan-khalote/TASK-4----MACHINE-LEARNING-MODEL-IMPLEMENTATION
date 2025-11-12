
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
# Using a sample dataset from sklearn or a simple CSV for demo
data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", 
                   sep='\t', header=None, names=['label', 'message'])

# Step 3: Data Preprocessing
data['label_num'] = data.label.map({'ham':0, 'spam':1})
X = data['message']
y = data['label_num']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Extraction
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Model Training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test_vec)

# Step 8: Evaluation
print("✅ Model Evaluation Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Sample Prediction
sample = ["Congratulations! You've won a free lottery ticket. Call now!"]
sample_vec = vectorizer.transform(sample)
print("\n📩 Sample Message Prediction:", model.predict(sample_vec))
