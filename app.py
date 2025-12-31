from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            BM_BLAST=float(request.form.get('BM_BLAST')),
            WBC=float(request.form.get('WBC')),
            ANC=float(request.form.get('ANC')),
            MONOCYTES=float(request.form.get('MONOCYTES')),
            HB=float(request.form.get('HB')),
            PLT=float(request.form.get('PLT')),
            Nmut=float(request.form.get('Nmut')), 
            CENTER=request.form.get('CENTER')       
        )
        
        # Convert to DataFrame (ready for the model)
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Launch Prediction
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # For Survival Analysis, the result is a "Risk Score".
        # Higher score = Higher risk (lower survival time).
        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)