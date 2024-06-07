from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the preprocessor and the model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('best_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Preprocess the input data
        df_processed = preprocessor.transform(df)

        # Make prediction
        prediction = model.predict(df_processed)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
