
import numpy as np
import pandas as pd
import os

import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#------------------------------
# 1. Limpieza de datos
#------------------------------
# Cargar archivo Excel
df = pd.read_excel("../data/raw/Adidas US Sales Datasets.xlsx",skiprows=4)
# Borro la 1º column (Unnamed)
df = df.drop(df.columns[0], axis=1)
# Ruta destino con el nombre solicitado
output_path = "../data/processed/Adidas_US_Sales_procesed.csv"

# Exportar a CSV
df.to_csv(output_path, index=False)

#------------------------------
# 2. Feature Engineering
#------------------------------

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 1. Carga de datos
df = pd.read_csv('Adidas_US_Sales_procesed.csv')

# 2. Feature Engineering temporal
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
df['Month'] = df['Invoice Date'].dt.month
df['Year'] = df['Invoice Date'].dt.year
df['DayOfWeek'] = df['Invoice Date'].dt.dayofweek
df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 3. DICCIONARIOS DE CATEGORÍAS (Orden explícito)
retailer = ["Foot Locker", "West Gear", "Sports Direct", "Kohl's", "Amazon", "Walmart"]
region = ["West", "Northeast", "Midwest", "South", "Southeast"]
product_cats = ["Men's Street Footwear", "Men's Athletic Footwear", "Men's Apparel",
                "Women's Street Footwear", "Women's Apparel", "Women's Athletic Footwear"]
sales_method = ["Online", "Outlet", "In-store"]

# 4. Aplicar OneHotEncoder con las categorías explícitas
categorical_cols = ['Retailer', 'Region', 'Product', 'Sales Method']
ohe = OneHotEncoder(categories=[retailer, region, product_cats, sales_method], 
                    sparse_output=False, handle_unknown='ignore')

# Transformar y crear un DataFrame con las nuevas columnas dummy
ohe_encoded = ohe.fit_transform(df[categorical_cols])
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
df_ohe = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df.index)

# 5. Seleccionar columnas numéricas y unirlas con las dummy
numeric_cols = ['Price per Unit', 'Month', 'Year', 'DayOfWeek', 'Is_Weekend']
df_numeric = df[numeric_cols].copy()

# Unir todo en el DataFrame final de features (X)
X = pd.concat([df_numeric, df_ohe], axis=1)
y = df['Total Sales']

# 6. Guardado para los siguientes scripts
X.to_csv('../data/processed/X_preprocessed.csv', index=False)
y.to_csv('../data/processed/y_target.csv', index=False)

#------------------------------
# 3. Feature Engineering
#------------------------------


# 1. Carga de datos
df = pd.read_csv('Adidas_US_Sales_procesed.csv')

# 2. Feature Engineering temporal
df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
df['Month'] = df['Invoice Date'].dt.month
df['Year'] = df['Invoice Date'].dt.year
df['DayOfWeek'] = df['Invoice Date'].dt.dayofweek
df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

# 3. DICCIONARIOS DE CATEGORÍAS (Orden explícito)
retailer = ["Foot Locker", "West Gear", "Sports Direct", "Kohl's", "Amazon", "Walmart"]
region = ["West", "Northeast", "Midwest", "South", "Southeast"]
product_cats = ["Men's Street Footwear", "Men's Athletic Footwear", "Men's Apparel",
                "Women's Street Footwear", "Women's Apparel", "Women's Athletic Footwear"]
sales_method = ["Online", "Outlet", "In-store"]

# 4. Aplicar OneHotEncoder con las categorías explícitas
categorical_cols = ['Retailer', 'Region', 'Product', 'Sales Method']
ohe = OneHotEncoder(categories=[retailer, region, product_cats, sales_method], 
                    sparse_output=False, handle_unknown='ignore')

# Transformar y crear un DataFrame con las nuevas columnas dummy
ohe_encoded = ohe.fit_transform(df[categorical_cols])
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
df_ohe = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df.index)

# 5. Seleccionar columnas numéricas y unirlas con las dummy
numeric_cols = ['Price per Unit', 'Month', 'Year', 'DayOfWeek', 'Is_Weekend']
df_numeric = df[numeric_cols].copy()

# Unir todo en el DataFrame final de features (X)
X = pd.concat([df_numeric, df_ohe], axis=1)
y = df['Total Sales']

# 6. Guardado para los siguientes scripts
X.to_csv('../data/processed/X_preprocessed.csv', index=False)
y.to_csv('../data/processed/y_target.csv', index=False)


#------------------------------
# 4. MEJOR MODELO
#------------------------------


# 1. Carga de datos
X = pd.read_csv('X_preprocessed.csv')
y = pd.read_csv('y_target.csv')['Total Sales']

# 2. División en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔍 Iniciando optimización con Pipeline + GridSearchCV...\n")

# --- A. Linear Regression ---
pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
# LR no tiene hiperparámetros que tunear, pero lo evaluamos por consistencia
grid_lr = GridSearchCV(pipe_lr, param_grid={}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_lr.fit(X_train, y_train)

# --- B. Random Forest ---
pipe_rf = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(random_state=42))])
param_rf = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(pipe_rf, param_grid=param_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# --- C. XGBoost ---
pipe_xgb = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor(random_state=42))])
param_xgb = {
    'xgb__n_estimators': [50, 100],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__max_depth': [3, 5]
}
grid_xgb = GridSearchCV(pipe_xgb, param_grid=param_xgb, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_xgb.fit(X_train, y_train)

# --- D. SVR ---
# Se establece n_jobs=1 para evitar bloqueos (hangs) del kernel en Jupyter.
# Si el dataset es muy grande, considera eliminar 'rbf' o reducir los valores de 'C'.
pipe_svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
param_svr = {
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1],
    'svr__kernel': ['linear', 'rbf'] # 'linear' suele ser mucho más rápido
}
grid_svr = GridSearchCV(
    pipe_svr, 
    param_grid=param_svr, 
    cv=3, 
    scoring='neg_mean_squared_error', 
    n_jobs=1  # <-- CAMBIO CLAVE: Evita fallos de multiprocesamiento en Jupyter
)
print("🔄 Entrenando SVR (esto puede tomar unos momentos)...")
grid_svr.fit(X_train, y_train)

# --- E. KNN ---
# Se establece n_jobs=1 por la misma razón de estabilidad.
pipe_knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
param_knn = {
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
}
grid_knn = GridSearchCV(
    pipe_knn, 
    param_grid=param_knn, 
    cv=3, 
    scoring='neg_mean_squared_error', 
    n_jobs=1  # <-- CAMBIO CLAVE
)
print("🔄 Entrenando KNN...")
grid_knn.fit(X_train, y_train)

# 3. Evaluación de los Mejores Modelos
modelos_optimizados = {
    "Linear Regression": grid_lr.best_estimator_,
    "Random Forest": grid_rf.best_estimator_,
    "XGBoost": grid_xgb.best_estimator_,
    "SVR": grid_svr.best_estimator_,
    "KNN": grid_knn.best_estimator_
}

print("📊 Resultados de la Optimización:")
mejor_rmse = float('inf')
mejor_modelo_nombre = ""
mejor_modelo_obj = None

for nombre, modelo in modelos_optimizados.items():
    preds = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{nombre:20} | RMSE: {rmse:10.2f} | R²: {r2:.4f} | Mejores Params: {modelo.get_params()}")
    
    if rmse < mejor_rmse:
        mejor_rmse = rmse
        mejor_modelo_nombre = nombre
        mejor_modelo_obj = modelo

# 4. Guardado del Mejor Modelo para Producción
print(f"\n🏆 MEJOR MODELO SELECCIONADO: {mejor_modelo_nombre} con RMSE: {mejor_rmse:.2f}")
with open("../models/mejor_modelo_produccion.pkl", "wb") as f:
    pickle.dump(mejor_modelo_obj, f)
print("💾 Modelo guardado exitosamente como 'mejor_modelo_produccion.pkl'")

# 5. Predicción Final de Ejemplo
nueva_entrada = X_test.iloc[[0]]
prediccion_final = mejor_modelo_obj.predict(nueva_entrada)
print(f"\n🔮 Predicción final de ejemplo: ${prediccion_final[0]:.2f} (Valor real en test: ${y_test.iloc[0]:.2f})")