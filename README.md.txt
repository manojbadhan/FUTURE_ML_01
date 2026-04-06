# 📊 Sales Forecasting using Time-Series (Prophet)

## 📌 Project Overview
This project builds a **sales forecasting system** using historical retail data.  
It uses **time-series analysis** to predict future sales and provide business insights.

The model helps businesses make better decisions related to:
- Inventory planning
- Demand forecasting
- Staffing
- Financial planning

---

## 🎯 Objective
To predict **future sales for the next 30 days** using past data and present results in a clear, business-friendly way.

---

## 📂 Dataset
- Superstore Sales Dataset
- Contains transactional data including:
  - Order Date
  - Sales
  - Product & customer details

---

## ⚙️ Project Workflow

### 1️⃣ Data Preparation
- Converted `Order Date` into datetime format
- Aggregated daily sales
- Renamed columns for time-series modeling

### 2️⃣ Exploratory Data Analysis
- Visualized sales trends over time
- Identified patterns in demand

### 3️⃣ Model Building
- Used **Facebook Prophet** for forecasting
- Captured:
  - Trend 📈
  - Weekly seasonality 📅
  - Yearly seasonality 📊

### 4️⃣ Forecasting
- Predicted sales for the **next 30 days**

### 5️⃣ Model Evaluation
- Used **Mean Absolute Error (MAE)** to measure accuracy

---

## 📊 Results

- Generated future sales predictions
- Identified demand fluctuations
- MAE used to evaluate model performance

---

## 📈 Visualizations
- Sales trend over time
- Forecast graph
- Trend & seasonality components

---

## 💼 Business Insights

- Sales show clear weekly patterns
- Certain days have consistently higher demand
- Forecast helps optimize:
  - Inventory management
  - Supply chain planning
  - Staffing decisions

---

## 🛠 Tech Stack

- Python
- Pandas
- Matplotlib
- Prophet
- Scikit-learn

---

## 📁 Project Structure
Sales-Forecasting/
│── data/
│ └── train.csv
│── sales_forecasting_project.py
│── sales_forecast_model.pkl
│── requirements.txt
│── README.md

---

## 📌 Key Features

✔ Time-series forecasting  
✔ Trend & seasonality analysis  
✔ Future sales prediction  
✔ Visualization for business understanding  
✔ Model evaluation (MAE)  
✔ Model saving  

---

## 💡 Business Impact

This model enables businesses to:
- Predict future demand accurately
- Reduce overstocking or stockouts
- Improve operational efficiency
- Make data-driven decisions

---

## 👨‍💻 Author

**Manoj Badhan**  
BTech AI & Robotics Engineering  

---

## 🚀 Future Improvements

- Deploy as a web app (Streamlit)
- Add Power BI dashboard
- Use advanced models (ARIMA, LSTM)