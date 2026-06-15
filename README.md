

# Predicción Venta ADIDAS en Estados Unidos
Proyecto orientado a la estimación de ventas de productos de Adidas en EE.UU. mediante Machine Learning.  
El objetivo es hacer una previsión de ventas lo mas precisa posible para mejorar la producción, reducir el lead time con los retailers, evitar roturas de stock de productos, hacer campañas de marketing orientadas a productos estratégicos de mercado.


## 📌 Alcance del Proyecto.
Este repositorio recoge una ruta completa de data science aplicado a una prevision de ingreso de ventas:
>- Procesamiento y limpieza de grandes volúmenes de datos.
>- Análisis exploratorio de variabels.
>- Definición avanzada de la Traget objetivo.
>- Entrenamiento y comparación de modelos predictivos.
>- Selección del mejor modelo predictivo
>- El enfoque productivo de la aplicaión desarrollada.


## 🧠 Enfoque Metodológico:
1️⃣ Analizar las tendencias de las ventas.
>- Identificar productos más exitosos.
>- Identificar el mejor retailer.
>- Identificar el estado que más vende.
>- Identificar el mejor canal de ventas.  

2️⃣ Análisis Exploratorio de Datos (EDA)
>- Dataset inicial: ~100.000 registros.
>- Limpieza avanzada.
>- Análisis visual.  

3️⃣ Ingeniería de Características (Features)
>- Transformación de variables.
>- Eliminacion de variables relacionadas.  

4️⃣ Definición del Target.
>- Total Sales.

5️⃣ Ruta y Validación.  
Modelos evaluados:
>- Regresión lineal.  
>- Random Forest optimizado.  
>- XGBoost optimizado.  
>- SVR optimizado.  
>- KNN optimizado.  
 
Metricas de modelo optimo:
>- MAE.  
>- RMSE.  
>- R²  


## 📎 Nota Final.
Este proyecto está diseñado como demostrador técnico de capacidades en Data Science en el entorno financiero y previsional de venta de productos. 
Si lo estás revisando desde un punto de vista profesional, el enfoque y las decisiones metodológicas están pensadas para facilitar un despliegue futuro en entorno productivo.


## 🛠️Estructura repositorio.

**Estructura** desarrollo del proyecto: adquisición de datos, limpieza, EDA, feature engineering, modelado de datos, iteración de modelos, evaluación de modelos, interpretación de modelos, impacto en negocio.

```
|-- nombre_proyecto_final_ML
    |-- data
    |   |-- raw
    |        |-- dataset.csv
    |        |-- ...
    |   |-- processed
    |
    |-- notebooks
    |   |-- 01_Fuentes.ipynb
    |   |-- 02_LimpiezaEDA.ipynb
    |   |-- 03_Entrenamiento_Evaluacion.ipynb
    |   |-- ...
    |
    |-- src
    |   |-- data_processing.py
    |   |-- training.py
    |   |-- evaluation.py
    |   |-- ...
    |
    |-- models
    |   |-- trained_model.pkl
    |   |-- model_config.yaml
    |   |-- ...
    |
    |-- app_streamlit
    |   |-- app.py
    |   |-- requirements.txt
    |   |-- ...
    |
    |-- docs
    |   |-- negocio.ppt
    |   |-- ds.ppt
    |   |-- memoria.md
    |   |-- ...
    |
    |
    |-- README.md

```


