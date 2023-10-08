from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__,template_folder='template')
preprocessor = joblib.load('preprocessor.joblib')
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weather = request.form['weather']
        season = request.form['season']
        previous_sales = float(request.form['previous_sales'])

        # Encode categorical features
        input_features = pd.DataFrame([[weather, season, previous_sales]], columns=['weather', 'season', 'previous_sales'])
        input_features_encoded = preprocessor.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features_encoded)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
