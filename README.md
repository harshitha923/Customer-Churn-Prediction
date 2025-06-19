# 🔁 Customer Churn Prediction and Behavior Analysis with Streamlit Interface

This project is a modular and interactive machine learning pipeline designed to predict customer churn using advanced analytics and a user-friendly Streamlit dashboard. It includes data preprocessing, classification, clustering, outlier detection, association rule mining, and an integrated chatbot for user engagement.

---

## 🚀 Features

- 📊 **Preprocessing**: Cleans and transforms raw customer data for model readiness.
- 🔎 **Outlier Detection**: Identifies anomalies in customer behavior.
- 🎯 **Classification**: Predicts whether a customer is likely to churn using ML algorithms.
- 🧠 **Clustering**: Groups similar customers based on behavior and demographics.
- 🧱 **Association Rule Mining**: Uncovers hidden patterns and relationships in customer attributes.
- 💬 **Chatbot**: Interact with a conversational AI chatbot built using Google’s Gemini API, capable of generating natural, contextual replies to customer queries.
- 🌐 **Streamlit Web App**: Frontend UI for interacting with the pipeline.

---

## 🛠️ Technologies Used

- Python 3.x
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- ML Algorithms: Logistic Regression, Decision Trees, KMeans
- Association Rule Mining: MLxtend
---

## 📦 Installation

1. **Clone the repository**

   Run:
   
     git clone https://github.com/harshitha923/customer-churn-prediction.git
     cd customer-churn-prediction
3. **Install required libraries**

Run:

    pip install -r requirements.txt
    
Create requirements.txt if not already present:
   
      streamlit
      pandas
      numpy
      scikit-learn
      matplotlib
      seaborn
      mlxtend

## ▶️ Running the Application
Launch the Streamlit web app locally:

    streamlit run app.py

You’ll be able to:

Upload your dataset

Visualize data distributions and correlations

Detect outliers and segment customers

Predict churn probabilities

View classification metrics (accuracy, F1, confusion matrix)

Chat with a basic support bot

## 📊 Machine Learning Techniques
Supervised Learning: Logistic Regression, Random Forests

Unsupervised Learning: KMeans Clustering, Outlier Detection

Association Mining: Apriori for rules between customer attributes

Chatbot Interaction: Simulated FAQ-style customer assistant

## 📌 Use Cases
Telecom and Subscription services

Customer Relationship Management (CRM)

Business Intelligence dashboards

EdTech / SaaS companies wanting churn insights

