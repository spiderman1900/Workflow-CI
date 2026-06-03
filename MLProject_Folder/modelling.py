import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # Set tracking URI ke local direktori Colab
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco_Churn_Eksperimen_Efendy")

    # Aktifkan Autolog
    mlflow.sklearn.autolog()
    print("--- MLflow Autolog Berhasil Diaktifkan ---")

    # Path dataset di Colab (Hasil Preprocessing Kriteria 1)
    data_path = "namadataset_preprocessing/data_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}.")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="Random_Forest_Base_Model"):
        print("Sedang melatih model Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        akurasi = accuracy_score(y_test, y_pred)
        
        print(f"\nModel selesai dilatih dengan Akurasi: {akurasi:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\n--- Semua parameter, metrik, dan artefak berhasil dicatat oleh MLflow! ---")

if __name__ == "__main__":
    train_model()
