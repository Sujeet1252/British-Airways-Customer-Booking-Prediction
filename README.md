# ✈️ British Airways Customer Booking Prediction

> **Predicting customer booking behavior using Machine Learning (Random Forest)**
>
> **British Airways Data Science Virtual Job Simulation**

---

## 📌 Project Overview

In the airline industry, understanding customer behavior before a booking is completed helps optimize marketing campaigns and improve conversion rates.

This project develops a **Machine Learning classification model** to predict whether a customer will complete a booking based on their travel preferences and booking details.

The project was completed as part of the **British Airways Data Science Virtual Job Simulation**, covering data exploration, preprocessing, predictive modeling, model evaluation, and business insights.

---

## 🎯 Business Objective

The primary objective is to predict:

**Will a customer complete the booking?**

By identifying the most influential booking factors, British Airways can:

- Improve targeted marketing campaigns
- Increase booking conversion rates
- Understand customer purchasing behavior
- Optimize promotional strategies

---

## 📂 Dataset Information

The dataset contains **50,000 customer booking records** with **14 features**, including:

| Feature | Description |
|----------|-------------|
| num_passengers | Number of passengers |
| sales_channel | Booking platform |
| trip_type | Round trip / One way |
| purchase_lead | Days before departure |
| length_of_stay | Trip duration |
| flight_hour | Flight departure hour |
| flight_day | Day of travel |
| route | Flight route |
| booking_origin | Customer country |
| wants_extra_baggage | Extra baggage purchase |
| wants_preferred_seat | Preferred seat selection |
| wants_in_flight_meals | Meal selection |
| flight_duration | Flight duration |
| booking_complete | Target Variable |

---

# ⚙️ Project Workflow

### 1️⃣ Data Exploration

- Dataset inspection
- Shape and information
- Statistical summary
- Missing value analysis
- Duplicate value detection

---

### 2️⃣ Data Preprocessing

- Removed duplicate records
- Label Encoding for categorical variables
- Feature selection
- Train-Test Split

---

### 3️⃣ Exploratory Data Analysis

Visualizations include:

- Booking Completion Distribution
- Correlation Heatmap
- Feature Importance
- Confusion Matrix
- ROC Curve

---

### 4️⃣ Machine Learning Model

Algorithm Used:

✅ Random Forest Classifier

Reasons:

- Handles categorical data efficiently
- Works well with mixed features
- Provides feature importance
- Robust against overfitting

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **84.9%** |
| ROC-AUC Score | **0.761** |
| Cross Validation | Performed (5-Fold) |

---

## 🔍 Key Findings

The most influential features affecting booking completion are:

1. Purchase Lead
2. Route
3. Flight Hour
4. Length of Stay
5. Booking Origin

These features contribute the most toward predicting customer booking behavior.

---

# 📊 Project Visualizations

## Booking Completion Distribution

Shows the distribution of completed and incomplete bookings.

---

## Correlation Heatmap

Displays relationships among numerical features.

---

## Confusion Matrix

Evaluates classification performance.

---

## ROC Curve

Illustrates the model's ability to distinguish between booking outcomes.

---

## Feature Importance

Ranks the most influential features used by the Random Forest model.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# 📁 Repository Structure

```
British-Airways-Customer-Booking-Prediction
│
├── Dataset
│   └── customer_booking.csv
│
├── Images
│   ├── booking_completion.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── Notebooks
│   └── Predictive_Modeling_of_Customer_Bookings.ipynb
│
├── Reports
│   └── British_Airways_Corporate_Presentation_Sujeet_Goswami.pptx
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/Sujeet1252/British-Airways-Customer-Booking-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
Notebooks/Predictive_Modeling_of_Customer_Bookings.ipynb
```

Run all cells.

---

# 📚 Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Data Visualization
- Feature Engineering
- Label Encoding
- Machine Learning
- Random Forest Classification
- Model Evaluation
- Business Insight Generation

---

# 💼 About This Project

This project was completed as part of the **British Airways Data Science Virtual Job Simulation**, where the objective was to build a predictive model that helps understand customer booking behavior and supports business decision-making.

---

# 👨‍💻 Author

**Sujeet Goswami**

MCA Student | Aspiring Data Scientist

GitHub: https://github.com/Sujeet1252

---

## ⭐ If you found this project useful, consider giving it a star!
