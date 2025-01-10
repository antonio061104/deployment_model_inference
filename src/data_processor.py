import pandas as pd
import numpy as np

def preprocessor_data(data, columns_to_impute):
    # Asegurarse de que data es un DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data debe ser un DataFrame")
    
    # Reemplazar ceros con NaN en las columnas especificadas
    data[columns_to_impute] = data[columns_to_impute].replace(0, np.nan)
    
    # Imputar valores faltantes (ejemplo simple usando la media)
    data.fillna(data.mean(), inplace=True)
    
    return data
