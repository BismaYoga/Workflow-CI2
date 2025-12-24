import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

def train_model():
    data_path = 'padi_preprocessing.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} tidak ditemukan!")
        return

    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Produksi'])
    y = df['Produksi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Set Eksperimen
    mlflow.set_experiment("Padi_Production_Tuning")

    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        rf = RandomForestRegressor(random_state=42)
        
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        print("Tuning dan Training Selesai menggunakan Autologging.")

if __name__ == "__main__":
    train_model()