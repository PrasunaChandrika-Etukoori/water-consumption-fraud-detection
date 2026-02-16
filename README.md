# 🚰 Water Consumption Fraud Detection

Machine learning–based system to identify **fraudulent water consumption behavior** using historical usage data.

---

## 🌍 Why this project?
Water supply companies lose significant revenue due to **non-technical losses**, where customers manipulate water meters to reduce billing amounts.  
Traditional manual inspection is **slow, expensive, and error-prone**.

This project automates fraud detection by applying **data mining and machine learning techniques** to customer water consumption patterns.

---

## 🎯 What does it do?
- Analyzes historical water usage data  
- Detects abnormal or fraudulent consumption behavior  
- Classifies customers as **Fraud** or **Non-Fraud**  
- Compares machine learning models to find the most effective one  

---

## 🧠 Approach
1. Collect and preprocess water consumption data  
2. Train supervised learning models  
3. Evaluate models using accuracy and recall metrics  
4. Use the best-performing model to predict fraud on new data  

---

## 📊 Dataset
- **Source:** Kaggle (Water Consumption Dataset)
- **Features:** Water usage values
- **Labels:**
  - `YES` → Fraud  
  - `NO` → Non-Fraud  

**Dataset Size**
- Total records: **8002**
- Training data: **6401**
- Testing data: **1601**

---

## 🔧 Data Preprocessing
- Convert categorical labels (`YES` / `NO`) to numeric values  
- Handle missing values  
- Normalize features using **Min-Max Scaling**  
- Split data into **70% training** and **30% testing**  

---

## 🤖 Machine Learning Models
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

---

## 📈 Model Performance

| Algorithm | Accuracy |
|----------|----------|
| SVM | ~90% |
| KNN | ~72% |

✅ **SVM outperformed KNN**, making it more suitable for fraud detection in this use case.

---

## 🖥️ Application Features
- Upload water consumption dataset  
- Automatic data preprocessing  
- Train and evaluate SVM and KNN models  
- Visualize accuracy comparison  
- Predict fraud for new test data  

The application is built using a **Python Tkinter GUI** for ease of use.

---

## ▶️ How to Run

### Option 1: Windows (Recommended)
1. Double-click `run.bat`
2. Upload the training dataset
3. Click **Preprocess Dataset**
4. Run **SVM** or **KNN**
5. Upload test dataset to predict fraud

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Tkinter (GUI)  
- Anaconda Environment  

---

## ✅ Conclusion
This project demonstrates how **machine learning models** can effectively detect fraudulent water consumption behavior.  
Among the tested models, **SVM achieved higher accuracy**, making it the preferred choice for this application.

---

📌 *Academic project (2021)
