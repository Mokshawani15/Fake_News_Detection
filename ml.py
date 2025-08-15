import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load datasets
data_fake1 = pd.read_csv("C:/Users/Moksha's Lappy/Desktop/GFG/Fake.csv")
data_true1 = pd.read_csv("C:/Users/Moksha's Lappy/Desktop/GFG/True.csv")
data_fake2 = pd.read_csv("C:/Users/Moksha's Lappy/Desktop/GFG/clean_fake_facts_5000.csv")
data_true2 = pd.read_csv("C:/Users/Moksha's Lappy/Desktop/GFG/clean_true_facts_5000 (2).csv")

# Ensure consistent columns
data_fake2.columns = data_fake1.columns  
data_true2.columns = data_true1.columns  

common_columns = list(set(data_fake1.columns) & set(data_fake2.columns))
data_fake1 = data_fake1[common_columns]
data_fake2 = data_fake2[common_columns]

common_columns = list(set(data_true1.columns) & set(data_true2.columns))
data_true1 = data_true1[common_columns]
data_true2 = data_true2[common_columns]

# Merge and shuffle datasets
merged_fake = pd.concat([data_fake1, data_fake2], ignore_index=True).sample(frac=1).reset_index(drop=True)
merged_true = pd.concat([data_true1, data_true2], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Labeling fake and true news
merged_fake["class"] = 0  
merged_true["class"] = 1  

# Extract last 10 rows for manual testing
data_fake_manual_testing = merged_fake.tail(10)
data_true_manual_testing = merged_true.tail(10)

# Remove last 10 rows from the datasets
data_fake = merged_fake.iloc[:-10]
data_true = merged_true.iloc[:-10]

# Merge fake and true datasets
data = pd.concat([data_fake, data_true], axis=0)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", '', text)
    return text

# Preprocess text data
data['text'] = data['text'].fillna('').apply(wordopt)

# Splitting into features and labels
x = data['text']
y = data['class']

# Splitting dataset into training (75%) and testing (25%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression Model
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)

print("\nLogistic Regression Model:")
print("Accuracy:", LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))

# Decision Tree Model
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)

print("\nDecision Tree Model:")
print("Accuracy:", DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))

# XGBoost Model
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(xv_train, y_train)
pred_xgb = xgb.predict(xv_test)

print("\nXGBoost Model:")
print("Accuracy:", xgb.score(xv_test, y_test))
print(classification_report(y_test, pred_xgb))

# Function to predict Fake or True news
def output_label(text, model):
    """Predict whether the news is Fake or True."""
    transformed_text = vectorization.transform([text])
    prediction = model.predict(transformed_text)
    return "Fake News" if prediction == 0 else "True News"

# Function to generate confidence score
def generate_confidence_score(news_text, xgb, DT, LR):
    """Generate a confidence score for a news article."""
    transformed_text = vectorization.transform([news_text])
    
    # Get predictions from each model
    pred_xgb = xgb.predict(transformed_text)[0]
    pred_dt = DT.predict(transformed_text)[0]
    pred_lr = LR.predict(transformed_text)[0]
    
    # Count the number of models predicting True News
    true_count = sum([pred_xgb, pred_dt, pred_lr])
    
    # Confidence Score Calculation (Average confidence)
    confidence_score = (xgb.score(xv_test, y_test) + DT.score(xv_test, y_test) + LR.score(xv_test, y_test)) / 3 * 100
    
    # Adjust score based on number of models predicting True
    if true_count == 3:
        return f"High Confidence: {confidence_score:.2f}%"
    elif true_count == 2:
        return f"Moderate Confidence: {confidence_score:.2f}%"
    elif true_count == 1:
        return f"Low Confidence: {confidence_score:.2f}%"
    else:
        return "Fake News (0% confidence)"

# Function for manual testing
def manual_testing(news_text):
    """Perform manual testing on an input news article."""
    print("\nPredictions:")
    print(f"Logistic Regression: {output_label(news_text, LR)}")
    print(f"Decision Tree: {output_label(news_text, DT)}")
    print(f"XGBoost: {output_label(news_text, xgb)}")

    confidence = generate_confidence_score(news_text, xgb, DT, LR)
    print(f"\nConfidence Score: {confidence}")

# Example Usage
news = input("\nEnter a news article for testing: ")
manual_testing(news)
