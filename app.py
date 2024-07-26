from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import sqlite3

app = Flask(__name__)

# Cargar modelo entrenado
model = joblib.load('prueba_modelos/gans_model.pkl')  # Cambiar la ruta aquí

# Conectar a la base de datos SQLite
def get_db_connection():
    conn = sqlite3.connect('data/clientes.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    conn = get_db_connection()
    clientes = conn.execute('SELECT DISTINCT cliente_id FROM transacciones').fetchall()
    conn.close()
    return render_template('index.html', clientes=clientes)

@app.route('/transacciones/<cliente_id>')
def get_transacciones(cliente_id):
    conn = get_db_connection()
    transacciones = conn.execute('SELECT * FROM transacciones WHERE cliente_id = ?', (cliente_id,)).fetchall()
    conn.close()
    return jsonify([dict(ix) for ix in transacciones])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cliente_id = data['cliente_id']
    nueva_transaccion = pd.DataFrame([data['transaccion']])
    
    # Obtener historial del cliente
    conn = get_db_connection()
    historial = pd.read_sql_query('SELECT * FROM transacciones WHERE cliente_id = ?', conn, params=(cliente_id,))
    conn.close()
    
    # Preprocesamiento de datos
    historial = preprocess_data(historial)
    nueva_transaccion = preprocess_data(nueva_transaccion, single=True)
    
    # Añadir nueva transacción al historial para predecir
    historial = historial.append(nueva_transaccion, ignore_index=True)
    
    # Predicción de anomalías
    predictions = model.predict(historial)
    
    # Determinar si la nueva transacción es anómala
    is_anomalous = predictions[-1] == 1
    
    return jsonify({'anomalous': is_anomalous})

def preprocess_data(data, single=False):
    # Implementar el preprocesamiento necesario para los datos
    # Normalización, selección de características, etc.
    # Ejemplo:
    if single:
        features = data[['monto', 'hora', 'tipo_transaccion', 'geolocalizacion']]
    else:
        features = data[['cliente_id', 'monto', 'hora', 'tipo_transaccion', 'geolocalizacion']]
    return features

if __name__ == '__main__':
    app.run(debug=True)
