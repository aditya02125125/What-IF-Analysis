import joblib
import pickle
import pandas as pd
import numpy as np

old_model = joblib.load("models/rf_model.pkl")
new_model = pickle.load(open("rf_optimized.pkl", "rb"))


df = pd.read_csv("cleaned_retail.csv", low_memory=False)


df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df["Revenue"] = df["Quantity"] * df["Price"]

top_products = df["StockCode"].value_counts().head(20).index
df = df[df["StockCode"].isin(top_products)]


df_old = pd.get_dummies(df, columns=["Country", "StockCode"])

X_old = df_old.drop(columns=[
    "Revenue",
    "Invoice",
    "Description",
    "InvoiceDate"
], errors="ignore")

df_new = pd.get_dummies(df, columns=["Country", "StockCode"])

X_new = df_new.drop(columns=[
    "Revenue",
    "Invoice",
    "Description",
    "Customer ID",
    "InvoiceDate"
], errors="ignore")

new_features = list(new_model.feature_names_in_)

for col in new_features:
    if col not in X_new.columns:
        X_new[col] = 0

X_new = X_new[new_features]

idx = 0

sample_old = X_old.iloc[[idx]]
sample_new = X_new.iloc[[idx]]

actual_value = df["Revenue"].iloc[idx]

old_pred = old_model.predict(sample_old)[0]
new_pred = new_model.predict(sample_new)[0]

print("\n===== Scenario Test =====")
print("Sample Index:", idx)

print("\nActual Revenue:", actual_value)

print("\nOld Model Prediction:", old_pred)
print("New Model Prediction:", new_pred)

old_error = abs(actual_value - old_pred)
new_error = abs(actual_value - new_pred)

print("\nOld Model Error:", old_error)
print("New Model Error:", new_error)

if new_error < old_error:
    print("\n✅ Optimized model is BETTER")
else:
    print("\n⚠️ Old model performed better on this sample")