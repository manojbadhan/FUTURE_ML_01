# =====================================
# 📦 1. Import Libraries
# =====================================
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
import pickle


# =====================================
# 📂 2. Load Dataset
# =====================================
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "train.csv")

    df = pd.read_csv(file_path)
    print("INFO: Dataset loaded successfully")

    return df


# =====================================
# 🧹 3. Data Preparation
# =====================================
def prepare_data(df):
    df = df.copy()

    print("Columns in dataset:", df.columns)

    # Convert date (DD/MM/YYYY handled)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    # Aggregate daily sales
    df = df.groupby('Order Date')['Sales'].sum().reset_index()

    # Rename for Prophet
    df = df.rename(columns={
        'Order Date': 'ds',
        'Sales': 'y'
    })

    print("INFO: Data prepared successfully")
    return df


# =====================================
# 📊 4. Visualization
# =====================================
def plot_sales(df):
    plt.figure()
    plt.plot(df['ds'], df['y'])
    plt.title("Sales Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()


# =====================================
# 🤖 5. Train Model
# =====================================
def train_model(df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(df)

    print("INFO: Model trained successfully")
    return model


# =====================================
# 🔮 6. Forecast Future
# =====================================
def forecast_future(model):
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    print("INFO: Future forecast generated")
    return forecast


# =====================================
# 📈 7. Plot Forecast
# =====================================
def plot_forecast(model, forecast):
    model.plot(forecast)
    plt.title("Sales Forecast (Next 30 Days)")
    plt.show()

    model.plot_components(forecast)
    plt.show()


# =====================================
# 📊 8. Evaluation
# =====================================
def evaluate_model(df, forecast):
    merged = df.merge(forecast[['ds', 'yhat']], on='ds')

    mae = mean_absolute_error(merged['y'], merged['yhat'])

    print(f"\nINFO: Model MAE = {mae:.2f}")


# =====================================
# 💾 9. Save Model
# =====================================
def save_model(model):
    with open("sales_forecast_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("INFO: Model saved successfully")


# =====================================
# 💼 10. Business Output
# =====================================
def business_output(forecast):
    future_sales = forecast[['ds', 'yhat']].tail(30)

    print("\n=== Forecast for Next 30 Days ===")
    print(future_sales)


# =====================================
# 🚀 11. Main Pipeline
# =====================================
def main():
    df = load_data()

    df = prepare_data(df)

    # Visualization
    plot_sales(df)

    model = train_model(df)

    forecast = forecast_future(model)

    plot_forecast(model, forecast)

    # Evaluation
    evaluate_model(df, forecast)

    # Business output
    business_output(forecast)

    # Save model
    save_model(model)


# =====================================
# ▶ Run
# =====================================
if __name__ == "__main__":
    main()