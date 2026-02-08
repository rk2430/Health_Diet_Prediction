# Healthy Diet Cost Prediction System 🥗📊

## 📌 Project Overview
The **Healthy Diet Cost Prediction System** is a Machine Learning–based web application that predicts the **cost of a healthy diet (PPP USD)** based on multiple economic and food-related factors.  
The system uses a **Linear Regression model** trained on real-world dietary cost data and provides predictions through a **Flask web interface**.

This project demonstrates the **complete ML lifecycle**:
- Data preprocessing
- Model training & evaluation
- Model persistence using joblib
- Web deployment using Flask

---

## 🎯 Objectives
- Predict the cost of a healthy diet accurately
- Allow users to input relevant parameters through a dashboard
- Demonstrate real-world ML deployment using Flask
- Follow best practices in ML preprocessing and model usage

---

## 🧠 Machine Learning Details
- **Algorithm Used**: Linear Regression
- **Libraries**: scikit-learn, pandas, joblib
- **Target Variable**:  
  `cost_healthy_diet_ppp_usd`
- **Input Features**:
  - Year
  - Annual Healthy Diet Cost (USD)
  - Vegetable Cost (PPP USD)
  - Fruit Cost (PPP USD)
  - Total Food Components Cost

---

## 🧹 Data Preprocessing
- Column name normalization
- Handling missing values using **mean imputation**
- Feature selection based on numerical relevance
- Train–test split (80% training, 20% testing)

---

## 🌐 Web Application Features
- User-friendly dashboard
- Manual input of all required features
- Real-time prediction
- Clean and aligned UI
- Error handling for invalid inputs

---

## 🏗️ Project Structure
