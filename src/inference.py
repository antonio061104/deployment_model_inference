import sys
import os
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_loader import load_model
from src.data_processor import preprocessor_data
import pandas as pd
import numpy as np
import joblib
def main():
    model_path = "models/trained_model_2025-01-09.joblib"
    try:
        model = joblib.load(model_path)
        print("paso la carga del modelo")
    except FileNotFoundError:
        print(f"Error loading model: {model_path} not found")
        sys.exit(1)

    input_data = {
        "age": 50,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 2
    }

    # Convertir input_data a DataFrame
    df = pd.DataFrame([input_data])
    
    # Verificar las columnas disponibles en input_data
    print("Columnas disponibles en input_data:", df.columns.tolist())
    
    # columns to use
    columns_to_use = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Preprocesar los datos
    try:
        preprocessed_data = preprocessor_data(data=df, columns_to_impute=columns_to_use)
        print("hizo el preprocesamiento con éxito")
    except KeyError as e:
        print(f"Error preprocessing data: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error preprocessing data: {e}")
        sys.exit(1)
    
    try:
        prediction = model.predict(preprocessed_data)
        print(f"Predictions: {prediction}")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
