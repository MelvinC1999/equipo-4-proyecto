from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
CORS(app)  # Permitir solicitudes CORS

# Cargar los modelos y scalers
gan_models_dir = 'gan_models'
scaler_dir = 'scalers'
top_clients = [1599382, 1598973]  # Ejemplo de clientes

def load_models_and_scalers(model_dir, scaler_dir, selected_clients):
    models = {}
    for client_id in selected_clients:
        generator_path = os.path.join(model_dir, f'generator_{client_id}.h5')
        discriminator_path = os.path.join(model_dir, f'discriminator_{client_id}.h5')

        if os.path.exists(generator_path) and os.path.exists(discriminator_path):
            generator = load_model(generator_path, compile=False)
            discriminator = load_model(discriminator_path, compile=False)

            scaler_path = os.path.join(scaler_dir, f'scaler_{client_id}.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print(f"Scaler para el cliente {client_id} no encontrado.")
                scaler = None

            models[client_id] = (generator, discriminator, scaler)
        else:
            print(f"Modelos para el cliente {client_id} no encontrados.")
            models[client_id] = (None, None, None)

    return models

gan_results = load_models_and_scalers(gan_models_dir, scaler_dir, top_clients)

# Placeholder for data_cleaned, load your cleaned data here
# This should be replaced with actual loading logic
data_cleaned = pd.read_csv('original_values_cleaned.csv') # Replace with your actual data path
original_values_cleaned = data_cleaned.copy()

def predict_anomaly_per_client(transaction, gan_results):
    client_id = transaction['Cod_Cliente']

    if client_id not in gan_results or gan_results[client_id][0] is None:
        return None

    generator, discriminator, scaler = gan_results[client_id]

    if scaler is None:
        return None

    transaction_df = pd.DataFrame([transaction])
    transaction_df['Horas'] = transaction_df['Horas'].apply(time_to_seconds)

    # Codificar las columnas categóricas como numéricas
    transaction_df['Transaccion'] = transaction_df['Transaccion'].astype('category').cat.codes

    # Agregar la columna Tiempo_Dia utilizando la misma lógica que en la celda 1
    transaction_df['Tiempo_Dia'] = transaction_df['Horas'].apply(categorize_time).astype('category').cat.codes

    # Guardar valores originales para la explicación
    original_values = transaction_df.copy()

    # Normalizar los valores de la transacción
    features = ['Valor', 'Tiempo_Dia', 'Dia_Semana', 'Mes', 'Dia_Mes', 'Transaccion']
    transaction_df[features] = scaler.transform(transaction_df[features])

    transaction_scaled = transaction_df[features].values

    noise = np.random.normal(0, 1, (1, 100))
    generated_data = generator.predict(noise)

    # Obtener los datos históricos del cliente
    client_data = data_cleaned[data_cleaned['Cod_Cliente'] == client_id][features].values

    if client_data.size == 0:
        return None

    # Calcula la distancia solo para la variable 'Valor'
    distance = np.abs(transaction_scaled[0, 0] - generated_data[0, 0])

    distances = np.abs(client_data[:, 0] - generated_data[:, 0])

    threshold = np.percentile(distances, 95)
    is_anomalous = distance > threshold

    # Obtener los valores históricos originales
    original_client_data = original_values_cleaned[original_values_cleaned['Cod_Cliente'] == client_id]

    max_historical_value = original_client_data['Valor'].max()
    most_frequent_hour = seconds_to_time(original_client_data['Horas'].mode()[0])
    most_frequent_day = original_client_data['Dia_Semana'].mode()[0]
    most_frequent_transaction = original_client_data['Transaccion'].mode()[0]
    most_frequent_value = original_client_data['Valor'].mode()[0]

    contributing_variable = 'Valor'
    variable_difference = distance

    # Desnormalizar el valor para la explicación
    real_value = scaler.inverse_transform(transaction_scaled)[0, 0]
    real_threshold = scaler.inverse_transform([[threshold] + [0] * (transaction_scaled.shape[1] - 1)])[0, 0]
    real_distance = np.abs(original_values['Valor'].values[0] - real_value)

    if is_anomalous:
        message = (
            f"La transacción con {transaction['Transaccion']} de valor {transaction['Valor']} realizada el {transaction['Dia_Mes']}/{transaction['Mes']} a las {transaction['Horas']} "
            f"ha sido clasificada como anómala. La principal razón es que la variable {contributing_variable} presenta una diferencia significativa de "
            f"{variable_difference} en comparación con los datos históricos del cliente {transaction['Cod_Cliente']}. "
            f"La distancia calculada es {real_distance}, que excede el umbral de {real_threshold}. "
            f"El valor máximo histórico para este cliente es {max_historical_value}. Los horarios más frecuentes son alrededor de las {most_frequent_hour} y los días más frecuentes son los días de la semana {most_frequent_day}. "
            f"El tipo de transacción más frecuente en montos altos es {most_frequent_transaction}. El valor más frecuente es {most_frequent_value}."
        )
    else:
        message = (
            f"La transacción con {transaction['Transaccion']} de valor {transaction['Valor']} realizada el {transaction['Dia_Mes']}/{transaction['Mes']} a las {transaction['Horas']} "
            f"ha sido clasificada como normal. La distancia calculada es {real_distance}, que está por debajo del umbral de {real_threshold}. "
            f"El valor máximo histórico para este cliente es {max_historical_value}. Los horarios más frecuentes son alrededor de las {most_frequent_hour} y los días más frecuentes son los días de la semana {most_frequent_day}. "
            f"El tipo de transacción más frecuente en montos altos es {most_frequent_transaction}. El valor más frecuente es {most_frequent_value}."
        )

    return message

def time_to_seconds(time_str):
    parts = time_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {time_str}")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def categorize_time(seconds):
    hour = seconds // 3600
    if 6 <= hour < 12:
        return 'Mañana'
    elif 12 <= hour < 18:
        return 'Tarde'
    else:
        return 'Noche'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = request.json
        app.logger.info(f'Received transaction: {transaction}')
        result = predict_anomaly_per_client(transaction, gan_results)
        if result is None:
            return jsonify({'error': 'Model or scaler not found for the given client.'}), 400
        app.logger.info(f'Result: {result}')
        return jsonify({'message': result})
    except ValueError as ve:
        app.logger.error(f'ValueError: {ve}')
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f'Error: {e}')
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
