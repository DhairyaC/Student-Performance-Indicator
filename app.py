import sys
from flask import Flask, request, render_template

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('index.html')
        else:
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity = request.form.get('race_ethnicity'),
                parental_level_of_education = request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course = request.form.get('test_preparation_course'),
                reading_score = request.form.get('reading_score'),
                writing_score = request.form.get('writing_score')
            )

            prediction_df = data.get_data_as_df()
            print(prediction_df)

            predict_pipeline = PredictPipeline()
            predicted_score = predict_pipeline.predict(prediction_df)

            return render_template('index.html', results=round(predicted_score[0], 2))
    
    except Exception as e:
        logging.info("Error is raised: {e}")
        raise CustomException(e, sys)
    
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)
        