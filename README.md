
**Overview**

Water supply companies face significant revenue loss due to fraudulent water consumption, where customers tamper with water meters to reduce billing amounts.
This project uses data mining and machine learning techniques to detect fraudulent water consumption behavior by analyzing historical consumption patterns.

The system classifies customers into Fraud and Non-Fraud categories using supervised learning models.


**Problem Statement**

There are two major types of water loss:

Technical Loss (TL): Leakage, transmission issues, system faults

Non-Technical Loss (NTL): Fraudulent consumption where delivered water is not billed

Manual inspection of water meters is time-consuming, costly, and error-prone.
This project automates fraud detection using machine learning classification models.



**Objectives**

Detect fraudulent water consumption behavior

Reduce non-technical losses

Assist water supply companies in fraud identification

Compare performance of different ML algorithms



**Proposed Solution**

The project uses Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) algorithms to classify customers based on their water consumption patterns.

The models are trained on historical consumption data and evaluated using accuracy and recall metrics.
The better-performing model is then used to predict fraud on new, unseen test data.



**Dataset**

Source: Kaggle (water consumption dataset)

Features: Water usage values

**Labels:**

YES → Fraud

NO → Non-Fraud

**Dataset size:**

Total records: 8002

Training set: 6401

Testing set: 1601



**Data Preprocessing**

The following preprocessing steps are applied:

Convert categorical labels (YES / NO) into numeric values

Handle missing values

Normalize features using Min-Max Scaling

Split data into training (70%) and testing (30%)



**Algorithms Used**

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)



**Model Performance**

Algorithm	Accuracy
SVM	         ~90%
KNN	         ~72%

SVM performs better than KNN for this dataset.



**Application Features**

Upload training dataset

Preprocess dataset automatically

Train SVM and KNN models

Visualize accuracy comparison

Upload test dataset

Predict whether water consumption is Fraud or Non-Fraud

The application is built using Python Tkinter GUI.



**How to Run the Project**

Option 1: Using Batch File (Windows)

Double-click run.bat

Upload the training dataset

Click Preprocess Dataset

Run SVM or KNN

Upload test dataset to predict fraud



**Tech Stack**

Python

Pandas

NumPy

Scikit-learn

Tkinter (GUI)

Anaconda Environment


**Conclusion**
This project demonstrates how machine learning models can effectively identify fraudulent behavior in
water consumption.
Among the tested models, SVM achieved higher accuracy, making it more suitable for fraud detection in
this use case.