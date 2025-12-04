import pickle
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# -----------------------------
# Load model + scaler
# -----------------------------
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':

        print("Received Form Data:", request.form)  # Debugging

        try:
            # Read form values safely
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            WS = float(request.form['WS'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])

            # Prepare for prediction
            input_data = [[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]]

            # Scale data
            scaled_data = standard_scaler.transform(input_data)

            # Predict
            prediction = ridge_model.predict(scaled_data)[0]

            return render_template('home.html', results=round(prediction, 4))

        except Exception as e:
            print("Error Occurred:", e)
            return render_template('home.html', results=f"Error: {str(e)}")

    return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
