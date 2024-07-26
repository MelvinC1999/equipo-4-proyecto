from sklearn.linear_model import LogisticRegression
import joblib

print("Creando el modelo de regresión logística dummy...")

# Crear un modelo de regresión logística dummy
model = LogisticRegression()

print("Entrenando el modelo con datos dummy...")

# Entrenar el modelo con datos dummy
X_dummy = [[0, 0], [1, 1]]
y_dummy = [0, 1]
model.fit(X_dummy, y_dummy)

print("Guardando el modelo en 'prueba_modelos/gans_model.pkl'...")

# Guardar el modelo en un archivo
joblib.dump(model, 'prueba_modelos/gans_model.pkl')

print("Modelo guardado correctamente.")