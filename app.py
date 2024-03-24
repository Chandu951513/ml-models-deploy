import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
 
app=Flask(__name__)
model=pickle.load(open('diabetes_train.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def Predict_api():
    data=request.json['data']
    print(data)
    k=np.array(list(data.values())).reshape(1,-1)
    data=scaler.transform(k)
    model.predict(data)
    print(data)
    return jsonify(data[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_out=scaler.transform(np.array(data).reshape(1,-1))
    out=model.predict(final_out)[0]
    return render_template("home.html",prediction_text="have diabetes is {}".format(out))




if __name__=="__main__":
    app.run(debug=True)
    
 