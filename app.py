from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the trained model and preprocessor
with open('model.pkl', 'rb') as model_file, open('preprocessor.pkl', 'rb') as preprocessor_file:
    model = pickle.load(model_file)
    preprocessor = pickle.load(preprocessor_file)

# Preprocess user input
def preprocess_data(input_data):
    input_df = pd.DataFrame([input_data])
    processed_data = preprocessor.transform(input_df)
    return processed_data

# Make prediction
def make_prediction(processed_input):
    prediction = model.predict(processed_input)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve user input, preprocess, make prediction, and render template
        age = request.form['age']
        gender = request.form['gender']
        stream = request.form['stream']
        internships = request.form['internships']
        cgpa = request.form['cgpa']
        hostel = request.form['hostel']
        history_of_backlogs = request.form['history_of_backlogs']

        user_input = {
            'Age': int(age),
            'Gender': gender,
            'Stream': stream,
            'Internships': int(internships),
            'CGPA': float(cgpa),
            'Hostel': int(hostel),
            'HistoryOfBacklogs': int(history_of_backlogs)
        }

        processed_input = preprocess_data(user_input)
        prediction = make_prediction(processed_input)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
