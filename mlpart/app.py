from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load the historical rainfall data
data_path = 'data/rainfall_data.csv'
rainfall_data = pd.read_csv(data_path)

# Preprocess the data
def preprocess_data(data):
    # Filter the columns to keep only the relevant ones
    data = data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
    # Convert data to long format
    data = data.melt(id_vars=["YEAR"], 
                     var_name="MONTH", 
                     value_name="RAINFALL")
    # Convert month names to numbers
    month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    data['MONTH'] = data['MONTH'].map(month_map)
    
    # Handle missing values in 'RAINFALL'
    data['RAINFALL'] = data['RAINFALL'].fillna(data['RAINFALL'].mean())
    return data

rainfall_data = preprocess_data(rainfall_data)

# Train a linear regression model
def train_model(data):
    X = data[['YEAR', 'MONTH']]
    y = data['RAINFALL']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model(rainfall_data)

# Define the prediction function
def predict_rainfall(year, month):
    prediction = model.predict(np.array([[year, month]]))
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        year = int(request.form['year'])
        month = int(request.form['month'])
        prediction = predict_rainfall(year, month)
        return render_template('result.html', prediction=prediction, year=year, month=month)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
