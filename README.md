 # Customer Churn Prediction using Artificial Neural Networks

This project implements a customer churn prediction system using an Artificial Neural Network (ANN). The model is built with Keras and TensorFlow and is deployed as an interactive web application using Streamlit.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Technologies Used](#-technologies-used)
- [Live Demo](#-live-demo)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)

## 🌟 Project Overview

The primary goal of this project is to predict whether a bank's customer will leave (churn) or stay with the bank. By analyzing various customer attributes, the model provides a probability of churn, which can help the bank take proactive measures to retain valuable customers. The core of the prediction system is a simple yet effective Artificial Neural Network.

## 💾 Dataset

The model was trained on the **Churn Modelling** dataset available on Kaggle. It contains 10,000 records of bank customers with 14 features.

- **Dataset Link:** [Churn Modelling on Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

Key features include:
- `CreditScore`
- `Geography` (France, Spain, Germany)
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (Target Variable: 1 if the customer churned, 0 otherwise)

## 🧠 Model Architecture

A sequential Artificial Neural Network (ANN) was constructed using the Keras library on top of TensorFlow. The architecture is as follows:

1.  **Input Layer:** Corresponds to the number of input features after preprocessing (e.g., 11 features after one-hot encoding).
2.  **Hidden Layer 1:** 6 neurons with the `ReLU` (Rectified Linear Unit) activation function.
3.  **Hidden Layer 2:** 6 neurons with the `ReLU` activation function.
4.  **Output Layer:** 1 neuron with the `Sigmoid` activation function, which outputs the probability of a customer churning.

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy

## 💻 Technologies Used

- **Data Processing:** Pandas, NumPy, Scikit-learn (for scaling and encoding)
- **Machine Learning:** TensorFlow, Keras
- **Web Framework:** Streamlit
- **Programming Language:** Python 3.8+

## 🚀 Live Demo

The prediction system is hosted as a web application where you can input customer details and get an instant churn prediction.

**🔗 Hosted App Link:** **[https://churn-prediction-system-vuxhcmxkaci4qgvdtyhhoo.streamlit.app/](https://churn-prediction-system-vuxhcmxkaci4qgvdtyhhoo.streamlit.app/)**

## ⚙️ Getting Started

To run this project locally, follow these steps:

**1. Download or Clone the repository:**

**Option A: Clone the repository (Recommended for developers)**
```bash
git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]