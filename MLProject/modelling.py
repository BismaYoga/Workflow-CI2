import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def train():
    data_file = 'padi_preprocessing.csv'
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} tidak ditemukan!")
        return

    df = pd.read_csv(data_file)
    X = df.drop(columns=['Produksi'])
    y = df['Produksi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if mlflow.active_run():
        mlflow.end_run()


    with mlflow.start_run(run_name="Padi_Retraining_Final"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Retraining Berhasil. MAE: {mae}")

if __name__ == "__main__":
    train()