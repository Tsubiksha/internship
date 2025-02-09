from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])

        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, 0]])  # Add 0 for 'Cabin'

        prediction = model.predict(input_data)

        result = "Survived" if prediction[0] == 1 else "Did Not Survive"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
