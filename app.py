from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model/price_model.pickle', 'rb'))

# Load columns from JSON
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    location_columns = data_columns[3:]  # All columns after bhk, bath, and total_sqft are locations

def get_estimated_price(location, total_sqft, bhk, bath):
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1  # If the location is not found, handle it gracefully
    
    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1  # Set the location in the array
    
    return round(model.predict([x])[0], 2)

@app.route('/')
def home():
    return render_template('app.html')  # Serve the HTML file

@app.route('/api/get_location_names', methods=['GET'])
def get_location_names():
    locations = location_columns
    return jsonify({
        'locations': locations
    })

@app.route('/api/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    location = request.form['location']

    estimated_price = get_estimated_price(location, total_sqft, bhk, bath)

    return jsonify({
        'estimated_price': estimated_price
    })

if __name__ == "__main__":
    app.run(debug=True)
