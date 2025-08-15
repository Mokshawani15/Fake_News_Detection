# Fake_News_Detection
# 📰 AI-Powered Fake News Detection System

## Overview
The **AI-Powered Fake News Detection System** is a **multi-layered AI-driven platform** designed to tackle misinformation by leveraging machine learning, NLP, fact-checking APIs, and credibility scoring.  
It ensures **accuracy**, **scalability**, and **reliability** in detecting manipulated or false news by combining multiple verification techniques.

---

## Problem Statement
The sudden rise of **misinformation** and **fake news** has turned into a global challenge, confusing the public and impacting **political**, **social**, and **financial** stability.  

Our **hybrid verification framework** ensures that if one method fails, others compensate, providing a **robust** and **explainable** credibility assessment.

---

## Proposed Solution

### **1. Machine Learning-Based Classification**
- **Model:** XGBoost
- Learns patterns from datasets of **real and fake news**.
- Enables automated misinformation detection with **high accuracy**.

### **2. Content Cross-referencing & Fact-checking**
- **External APIs:** Google Fact Check API, Media Bias Fact Check.
- Verifies claims against trusted news databases.
- Detects manipulated or misleading narratives.

### **3. Linguistic Analysis**
- **NLP-based sentiment analysis** using SpaCy and RoBERTa.
- **Stylometry methods** to detect author style changes.
- Identifies **biased** or **sensationalist** language.

### **4. BERT for Truthfulness Probability**
- **BERT (Bidirectional Encoder Representations from Transformers)** fine-tuned on labeled datasets.
- Classifies fake vs. real news and assigns a **truthfulness probability**.

---

## Uniqueness & Innovations
- Uses **four different verification methods**, making it stronger and less error-prone.
- Generates a **credibility score (0–100%)** instead of a simple true/false label.
- **Dynamically updates** its fact database for up-to-date detection.
- **Refines** credibility scores based on **user interactions** and **expert feedback**.

---

## Methodology

1. **Data Collection & Preprocessing**
   - Collect news articles from sources like Kaggle.
   - Clean and normalize the text data.

2. **Feature Extraction**
   - Use NLP techniques to extract features and identify entities.

3. **Model Training**
   - Train ML models using SpaCy, XGBoost, and BERT for classification and sentiment analysis.

4. **Cross-referencing**
   - Verify facts by querying news and fact-checking APIs.

5. **Error Detection**
   - Use RoBERTa to detect inconsistencies.

6. **Evaluation**
   - Calculate credibility scores using:
