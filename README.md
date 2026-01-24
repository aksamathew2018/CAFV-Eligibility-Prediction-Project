# 🚗 CAFV Eligibility Prediction

This project focuses on building an end-to-end **Machine Learning solution** to predict **Clean Alternative Fuel Vehicle (CAFV) eligibility** using a real-world electric vehicle dataset from Washington State. The goal is to classify whether a vehicle qualifies for CAFV incentives based on its technical and registration attributes.

---

## 📌 Project Objective
The objective of this project is to:
- Analyze electric vehicle data
- Identify key factors influencing CAFV eligibility
- Build and evaluate a classification model
- Deploy the model as an interactive web application

---

## 🧠 Machine Learning Approach

### 🔹 Data Preprocessing
- Cleaned and prepared the dataset by handling irrelevant and identifier columns
- Applied **Label Encoding** to categorical features such as County, City, Make, Model, EV Type, and Electric Utility
- Applied **StandardScaler** to numerical features including **Electric Range** and **Vehicle Age**

### 🔹 Feature Selection
- Used **Mutual Information** to measure non-linear dependency between features and the target
- Identified Electric Range as the most informative feature
- Analyzed feature relevance and detected **target leakage** due to regulatory rules embedded in the data

### 🔹 Model Building
- Trained and evaluated multiple classification models
- Selected **Random Forest Classifier** based on performance and robustness
- Used probability outputs (`predict_proba`) to estimate model confidence

---

## 🎯 Features Used
County
City
State
Postal Code
Make
Model
Electric Vehicle Type
Electric Range
Electric Utility
Age
## 🎯 Target Variable
0 → Clean Alternative Fuel Vehicle Eligible
1 → Eligibility unknown as battery range has not been researched

---

## 🖥️ Streamlit Web Application
The trained model is deployed as a **Streamlit web app** that allows users to:
- Enter vehicle details through a clean UI
- Receive CAFV eligibility prediction
- View a human-readable explanation of the result
- See the model’s confidence score

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/cafv-eligibility-prediction.git
cd cafv-eligibility-prediction
 Project Structure
cafv-eligibility-prediction/
│── app.py                 # Streamlit application
│── rf_best.pkl            # Trained Random Forest model
│── training_df.pkl        # Training dataset
│── requirements.txt       # Project dependencies
│── README.md              # Project documentation


✨ Key Learnings


End-to-end ML pipeline development


Feature engineering and preprocessing


Feature selection using Mutual Information


Identification of target leakage


Model evaluation and probability interpretation


Deployment using Streamlit and GitHub



👩‍💻 Author
Aksa Mathew
Machine Learning & Data Science Enthusiast
Kerala, India

⭐ Acknowledgement
This project is developed as part of hands-on learning in machine learning and model deployment. If you find it useful, feel free to ⭐ star the repository.

---

