import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

''' EDA '''
df_raw = pd.read_csv("../data/abalone.data")
df_raw.columns = ['Sex', 'Lenght', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Class']
print(df_raw.head())

print(df_raw.describe())

df_anomaly = df_raw[df_raw['Height'] == 0]
print(df_anomaly)

df_sex_i = df_raw[df_raw['Sex'] == 'I']
print(df_sex_i.count())

anomaly_index = df_raw[df_raw['Height'] == 0].index
df_data = df_raw.drop(anomaly_index)
print(df_data.describe())

df_data = pd.get_dummies(df_data, columns = ['Sex'], dtype = float)
print(df_data)

graph_columns = ['Lenght', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Class']
pd.plotting.scatter_matrix(df_data[graph_columns], figsize = (18, 18))

corr_matrix = df_data.corr()
corr_clase = corr_matrix['Class'].sort_values(ascending = False)
print(corr_clase)

''' Estandarización de los datos '''

x_features = df_data[['Lenght', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']]
x_mean = np.mean(x_features, axis = 0)
x_std = np.std(x_features, axis = 0)
x_std[x_std == 0] = 1e-8
x_scaled = (x_features - x_mean) / x_std
print(x_scaled.describe())

''' Separación de los datos '''

# Convertimos el data set a arreglos de numpy
X = x_scaled.to_numpy()
Y = df_data['Class'].to_numpy()

# Paso 1: Dividir en 70% entrenamiento y 30% temporal
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size = 0.30, random_state = 42
)

# Paso 2: Dividir el 30% temporal en 15% validación y 15% prueba
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size = 0.50, random_state = 42
)

print("Train (70%):", X_train.shape, Y_train.shape)
print("Validation (15%):", X_val.shape, Y_val.shape)
print("Test (15%):", X_test.shape, Y_test.shape)

''' Modelo de regresión '''

model = SGDRegressor(learning_rate = 'constant', eta0 = 0.01, penalty = None, random_state = 42)

epochs = 1000
cost_history = []

# Punto de partida sin entrenar
cost_history.append(np.mean((np.zeros(X_train.shape[0]) - Y_train) ** 2))

for epoch in range(epochs):
    model.partial_fit(X_train, Y_train)
    Y_hat = model.predict(X_train)
    cost_history.append(np.mean((Y_hat - Y_train) ** 2))

w = model.coef_
b = model.intercept_[0]

plt.figure(figsize = (8, 5))
plt.plot(range(epochs + 1), cost_history, color = 'blue', label = 'Train (SGDRegressor)')
plt.title('Curva de aprendizaje del modelo con scikit-learn')
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.ylim(0, 10)
plt.xlim(0, 10)
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.show()
print('\nLoss más bajo train')
print(np.min(cost_history))

model = SGDRegressor(learning_rate = 'constant', eta0 = 0.01, penalty = None, random_state = 42)

epochs = 1000
cost_history = []
val_cost_history = []

cost_history.append(np.mean((np.zeros(X_train.shape[0]) - Y_train) ** 2))
val_cost_history.append(np.mean((np.zeros(X_val.shape[0]) - Y_val) ** 2))

for epoch in range(epochs):
    model.partial_fit(X_train, Y_train)
    Y_hat = model.predict(X_train)
    cost_history.append(np.mean((Y_hat - Y_train) ** 2))

    Y_val_hat = model.predict(X_val)
    val_cost_history.append(np.mean((Y_val_hat - Y_val) ** 2))

plt.figure(figsize = (8, 5))
plt.plot(range(epochs + 1), cost_history, color = 'blue', label = 'Train')
plt.plot(range(epochs + 1), val_cost_history, color = 'red', label = 'Validation')
plt.title('Curva de aprendizaje del modelo con scikit-learn')
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.ylim(0, 10)
plt.xlim(0, 10)
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.show()
print('\nLoss más bajo train')
print(np.min(cost_history))
print('\nLoss más bajo validation')
print(np.min(val_cost_history))

model = SGDRegressor(learning_rate = 'constant', eta0 = 0.01, penalty = None, random_state = 42)

epochs = 1000
cost_history = []
val_cost_history = []
test_cost_history = []

cost_history.append(np.mean((np.zeros(X_train.shape[0]) - Y_train) ** 2))
val_cost_history.append(np.mean((np.zeros(X_val.shape[0]) - Y_val) ** 2))
test_cost_history.append(np.mean((np.zeros(X_test.shape[0]) - Y_test) ** 2))

for epoch in range(epochs):
    model.partial_fit(X_train, Y_train)
    Y_hat = model.predict(X_train)
    cost_history.append(np.mean((Y_hat - Y_train) ** 2))

    Y_val_hat = model.predict(X_val)
    val_cost_history.append(np.mean((Y_val_hat - Y_val) ** 2))

    Y_test_hat = model.predict(X_test)
    test_cost_history.append(np.mean((Y_test_hat - Y_test) ** 2))

# Guardamos parámetros para las siguientes celdas
w = model.coef_
b = model.intercept_[0]

plt.figure(figsize = (8, 5))
plt.plot(range(epochs + 1), cost_history, color = 'blue', label = 'Train')
plt.plot(range(epochs + 1), val_cost_history, color = 'red', label = 'Validation')
plt.plot(range(epochs + 1), test_cost_history, color = 'green', label = 'Test')
plt.title('Curva de aprendizaje del modelo con scikit-learn')
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.legend()
plt.ylim(0, 10)
plt.xlim(0, 10)
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.show()
print('\nLoss más bajo train')
print(np.min(cost_history))
print('\nLoss más bajo validation')
print(np.min(val_cost_history))
print('\nLoss más bajo test')
print(np.min(test_cost_history))

''' Coeficientes del modelo '''

print("\nCoeficientes del Modelo (Pesos) ")
print(f"Sesgo (Edad base estimada cuando x = 0): {b:.4f}")
for i, j in enumerate(w):
    print(f"Feature {i+1} (w{i+1}): {j:.4f}")

''' Métricas '''

# Predicciones finales usando el modelo entrenado con scikit-learn
Y_pred = model.predict(X_test)

# 1. Error Cuadrático Medio (MSE)
mse_test = mean_squared_error(Y_test, Y_pred)

# 2. Coeficiente de Determinación (R²)
r2 = r2_score(Y_test, Y_pred)

print("Métricas del modelo\n")
print(f"Mean Square Error: {mse_test:.4f}")
print(f"R²: {r2:.4f}")
print(f"Años de diferencia promedio alejada de la predicción: {np.sqrt(mse_test):.4f}")

''' Regularización L2 '''

lambda_reg = 0.1
model_ridge = Ridge(alpha=lambda_reg)
model_ridge.fit(X_train, Y_train)

mse_train_ridge = mean_squared_error(Y_train, model_ridge.predict(X_train))
mse_val_ridge   = mean_squared_error(Y_val,   model_ridge.predict(X_val))
mse_test_ridge  = mean_squared_error(Y_test,  model_ridge.predict(X_test))
r2_ridge        = r2_score(Y_test,            model_ridge.predict(X_test))

print("Métricas del modelo con regularización L2 usando Ridge\n")
print(f"MSE Train:      {mse_train_ridge:.4f}")
print(f"MSE Validation: {mse_val_ridge:.4f}")
print(f"MSE Test:       {mse_test_ridge:.4f}")
print(f"R²:             {r2_ridge:.4f}")
print(f"RMSE:           {np.sqrt(mse_test_ridge):.4f}")

print("\nCoeficientes Ridge")
print(f"Sesgo: {model_ridge.intercept_:.4f}")
for i, j in enumerate(model_ridge.coef_):
    print(f"Feature {i+1}: {j:.4f}")

''' Diferentes alphas para la regularización L2 '''

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
mse_train_list, mse_val_list, mse_test_list = [], [], []

for a in alphas:
    m = Ridge(alpha=a)
    m.fit(X_train, Y_train)
    mse_train_list.append(mean_squared_error(Y_train, m.predict(X_train)))
    mse_val_list.append(mean_squared_error(Y_val,     m.predict(X_val)))
    mse_test_list.append(mean_squared_error(Y_test,   m.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(alphas, mse_train_list, color='blue',  label='Train',      marker='o')
plt.plot(alphas, mse_val_list,   color='red',   label='Validation', marker='o')
plt.plot(alphas, mse_test_list,  color='green', label='Test',       marker='o')
plt.xscale('log')
plt.title('Efecto de la regularización L2 (Ridge) sobre el MSE')
plt.xlabel('Alpha (fuerza de regularización)')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

''' Ajuste de hiperparámetros '''

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
resultados = []

for lr in learning_rates:
    m = SGDRegressor(learning_rate='constant', eta0=lr, penalty=None,
                     random_state=42, max_iter=1000, tol=None)
    m.fit(X_train, Y_train)
    mse_tr = mean_squared_error(Y_train, m.predict(X_train))
    mse_v  = mean_squared_error(Y_val,   m.predict(X_val))
    mse_te = mean_squared_error(Y_test,  m.predict(X_test))
    r2_te  = r2_score(Y_test,            m.predict(X_test))
    resultados.append((lr, mse_tr, mse_v, mse_te, r2_te))
    print(f"lr={lr:.3f} | MSE train={mse_tr:.4f} | val={mse_v:.4f} | test={mse_te:.4f} | R²={r2_te:.4f}")