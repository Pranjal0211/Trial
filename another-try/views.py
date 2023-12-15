from flask import Blueprint, render_template, request
from pgmpy.inference import VariableElimination
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from app import app

#views = Blueprint(__name__, "views")

#load the dataset and build the bayesian network 
heartDisease = pd.read_csv('./sample_data/heart_disease_uci_copy10.csv')

model = BayesianNetwork([('age', 'trestbps'), ('sex', 'fbs'), ('trestbps', 'chol'),
                         ('fbs', 'chol'), ('chol', 'num'), ('restecg', 'num'),
                         ('thalch', 'exang'), ('exang', 'cp'), ('cp', 'num')])


model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
HeartDisease_infer = VariableElimination(model)


# Define the route for the home page
#@views.route("/")
@app.route("/")
def home():
    return render_template("index.html")

#define the route fort the diagnosis result page
@app.route("/diagnose", method=["POST"])
def diagnose():
    if request.method == 'POST':
        age = int(request.form['age'])
        trestbps = int(request.form['trestbps'])
        sex = int(request.form['sex'])
        fbs = int(request.form['fbs'])
        chol = int(request.form['chol'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        cp = int(request.form['cp'])

        # Perform inference
        query_result = HeartDisease_infer.query(variables=['num'],
                                               evidence={'age': age, 'trestbps': trestbps, 'sex': sex,
                                                         'fbs': fbs, 'chol': chol, 'restecg': restecg,
                                                         'thalch': thalach, 'exang': exang, 'cp': cp})
        
        result = query_result['num'].values[0]

        return render_template('result.html', result=result)

