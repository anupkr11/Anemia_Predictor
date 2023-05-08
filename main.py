from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')

def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    Gender = request.form.get('Gender')
    Hemoglobin = request.form.get('Hemoglobin')
    MCH = request.form.get('MCH')
    MCHC = request.form.get('MCHC')
    MCV = request.form.get('MCV')

    input_query = np.array([[Gender,Hemoglobin,MCH,MCHC,MCV]])

    Result = model.predict(input_query)[0]

    return jsonify({'Disease':str(Result)})

if __name__ == '__main__':
    app.run(debug=True)


