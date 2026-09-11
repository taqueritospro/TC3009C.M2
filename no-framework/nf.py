import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_raw = pd.read_csv("../data/abalone.data")
df_raw.columns = ['Sex', 'Lenght', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Class']
df_raw.head()

df_raw.describe()

df_anomaly = df_raw[df_raw['Height'] == 0]
df_anomaly

df_sex_i = df_raw[df_raw['Sex'] == 'I']
df_sex_i.count()

anomaly_index = df_raw[df_raw['Height'] == 0].index
df_data = df_raw.drop(anomaly_index)
df_data.describe()

df_data = pd.get_dummies(df_data, columns = ['Sex'], dtype = float)
df_data

graph_columns = ['Lenght', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Class']
pd.plotting.scatter_matrix(df_data[graph_columns], figsize = (18, 18))

corr_matrix = df_data.corr()
corr_clase = corr_matrix['Class'].sort_values(ascending = False)
corr_clase

x_features = df_data[['Lenght', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']]
x_mean = np.mean(x_features, axis = 0)
x_std = np.std(x_features, axis = 0)
x_std[x_std == 0] = 1e-8
x_scaled = (x_features - x_mean) / x_std
x_scaled.describe()

# Convertimos el data set a arreglos de numpy
X = x_scaled.to_numpy()
Y = df_data['Class'].to_numpy()
# Obtenemos las filas del data set y las mezclamos aleatoriamente
total_rows = X.shape[0]
np.random.seed(42)
random_index = np.random.permutation(total_rows)
# Reordenamos los índices del data set usando los aleatorios
X = X[random_index]
Y = Y[random_index]
# Hacemos el cálculo para dividir el data set en train, validation y test.
train_slice = int(0.7 * total_rows)
val_slice = int(0.85 * total_rows)
# Separamos el data set en sus respectivos conjuntos
X_train = X[:train_slice]
Y_train = Y[:train_slice]

X_val = X[train_slice:val_slice]
Y_val = Y[train_slice:val_slice]

X_test = X[val_slice:]
Y_test = Y[val_slice:]
# Revisamos que si se hayan hecho bien las separaciones
print("Train (70%):", X_train.shape, Y_train.shape)
print("Validation (15%):", X_val.shape, Y_val.shape)
print("Test (15%):", X_test.shape, Y_test.shape)

# Tamaño de los atributos
features_size = X_train.shape[1]

# Inicializamos los pesos y el sesgo en 0
w = np.zeros(features_size)
b = 0.0

# Learning rate que nos dice de cuanto en cuanto avanzamos
alpha = 0.01

# El número de iteraciones que haremos para entrenar el modelo
epochs = 1000

# El número de instancias para entrenar
n = X_train.shape[0]

# Historial de costo para la gráfica
cost_history = []

for epoch in range(epochs):
    # Función de hipótesis
    Y_hat = np.dot(X_train, w) + b

    # Función de costo
    loss = Y_hat - Y_train
    cost = (1 / (2 * n)) * np.sum(loss ** 2)
    cost_history.append(cost)

    # Gradiente descendiente
    dw = (1 / n) * np.dot(X_train.T, loss)
    db = (1 / n) * np.sum(loss)

    # Actualización de parámetros
    w = w - (alpha * dw)
    b = b - (alpha * db)

# Graficamos los resultados
plt.figure(figsize = (8, 5))
plt.plot(range(epochs), cost_history, color = 'blue')
plt.title('Curva de aprendizaje del modelo')
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.show()

# Ahora obtenemos el tamaño de validation
n_val = X_val.shape[0]

# Listas para guardar el historial de ambos costos
cost_history = []
val_cost_history = []

# Reiniciamos los parámetros
w = np.zeros(features_size)
b = 0.0

for epoch in range(epochs):
    # Para TRAIN
    # Nuestra función de hipótesis
    Y_hat = np.dot(X_train, w) + b

    # Nuestra función de costo
    loss = Y_hat - Y_train
    cost = (1 / (2 * n)) * np.sum(loss ** 2)
    cost_history.append(cost)

    # Para la VALIDATION
    # Nuestra función de hipótesis
    Y_val_hat = np.dot(X_val, w) + b

    # Nuestra función de costo
    loss_val = Y_val_hat - Y_val
    cost_val = (1 / (2 * n_val)) * np.sum(loss_val ** 2)
    val_cost_history.append(cost_val)

    # Gradiente descendiente que solo aplica para TRAIN
    # Ya que solo queremos evaluar con el conjunto VALIDATION
    dw = (1 / n) * np.dot(X_train.T, loss)
    db = (1 / n) * np.sum(loss)

    # Actualización de parámetros
    w = w - (alpha * dw)
    b = b - (alpha * db)

# Finalmente graficamos los resultados
plt.figure(figsize = (8, 5))
plt.plot(range(epochs), cost_history, color = 'blue', label = 'Train')
plt.plot(range(epochs), val_cost_history, color = 'red', label = 'Validation')
plt.title('Curva de aprendizaje del modelo')
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.show()

# Ahora obtenemos el tamaño de validation
n_val = X_val.shape[0]
n_test = X_test.shape[0]

# Listas para guardar el historial de ambos costos
cost_history = []
val_cost_history = []
test_cost_history = []

# Reiniciamos los parámetros
w = np.zeros(features_size)
b = 0.0

for epoch in range(epochs):
    # Para TRAIN
    # Nuestra función de hipótesis
    Y_hat = np.dot(X_train, w) + b

    # Nuestra función de costo
    loss = Y_hat - Y_train
    cost = (1 / (2 * n)) * np.sum(loss ** 2)
    cost_history.append(cost)

    # Para VALIDATION
    # Nuestra función de hipótesis
    Y_val_hat = np.dot(X_val, w) + b

    # Nuestra función de costo
    loss_val = Y_val_hat - Y_val
    cost_val = (1 / (2 * n_val)) * np.sum(loss_val ** 2)
    val_cost_history.append(cost_val)

    # Para TEST
    # Nuestra función de hipótesis
    Y_test_hat = np.dot(X_test, w) + b

    # Nuestra función de costo
    loss_test = Y_test_hat - Y_test
    cost_test = (1 / (2 * n_test)) * np.sum(loss_test ** 2)
    test_cost_history.append(cost_test)

    # Gradiente descendiente que solo aplica para TRAIN
    # Ya que solo queremos evaluar con el conjunto VALIDATION
    dw = (1 / n) * np.dot(X_train.T, loss)
    db = (1 / n) * np.sum(loss)

    # Actualización de parámetros
    w = w - (alpha * dw)
    b = b - (alpha * db)

# Finalmente graficamos los resultados
plt.figure(figsize = (8, 5))
plt.plot(range(epochs), cost_history, color = 'blue', label = 'Train')
plt.plot(range(epochs), val_cost_history, color = 'red', label = 'Validation')
plt.plot(range(epochs), test_cost_history, color = 'green', label = 'Test')
plt.title('Curva de aprendizaje del modelo')
plt.xlabel('Epochs')
plt.ylabel('Cost MSE')
plt.legend()
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.show()

print("\nCoeficientes del Modelo (Pesos) ")
print(f"Sesgo (Edad base estimada cuando x = 0): {b:.4f}")
for i, j in enumerate(w):
    print(f"Feature {i+1} (w{i+1}): {j:.4f}")