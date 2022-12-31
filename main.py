import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

FAST_API_PORT = os.environ.get('FAST_API_PORT') if os.environ.get('FAST_API_PORT') else 8000
FAST_API_HOST = os.environ.get('FAST_API_HOST') if os.environ.get('FAST_API_HOST') else "localhost"

app = FastAPI()

std_scaler =pickle.load(open('./models/std_scaler.pkl', 'rb'))
bag_model = pickle.load(open('./models/bagging_diabetes_model.pkl','rb'))
lgmb_model = pickle.load(open('./models/lgbm_diabetes_model.pkl','rb'))

class Infos(BaseModel):
    pregnancies : int
    glucose : int
    blood_pressure : int
    skin_thickness : int
    insulin : int
    bmi : float
    diabetes_pedigree_function : float
    age : int









def get_r1(dict_):
    if (int(dict_.get('age'))<=30) and (int(dict_.get('glucose'))<=120):
        return 0
    elif ((int(dict_.get('age'))>30) and (int(dict_.get('age'))<48)) and (int(dict_.get('glucose'))<=88):
        return 0
    elif (int(dict_.get('age'))) and (int(dict_.get('glucose'))<=142):
        return 0
    else:
        return 1

def get_r2(dict_):

    if (float(dict_.get('bmi'))<=30):
        return 0
    else:
        return 1

def get_r3(dict_):

    if (int(dict_.get('age'))<=27) and (int(dict_.get('pregnancies'))<=6):
        return 0
    elif (int(dict_.get('age'))>60) and (int(dict_.get('pregnancies'))>7.5):
        return 0
    else:
        return 1

def get_r4(dict_):

    if (int(dict_.get('glucose'))<=105) and (int(dict_.get('blood_pressure'))<=80):
        return 0
    elif (int(dict_.get('glucose'))<=105) and (int(dict_.get('blood_pressure'))>83):
        return 0
    else:
        return 1

def get_r5(dict_):

    if (int(dict_.get('skin_thickness'))<=20):
        return 0
    else:
        return 1

def get_r6(dict_):

    if (int(dict_.get('glucose'))<=105) and (float(dict_.get('bmi'))<=30):
        return 0
    elif (int(dict_.get('glucose'))<=105) and (float(dict_.get('bmi'))>41):
        return 0
    else:
        return 1

def get_r7(dict_):

    if int(dict_.get('insulin'))<200:
        return 0
    else:
        return 1

def get_r8(dict_):
    if int(dict_.get('blood_pressure'))<80:
        return 0
    else:
        return 1

def get_r9(dict_):
    if (float(dict_.get('bmi'))<30) and (int(dict_.get('skin_thickness'))<21):
        return 0
    elif (float(dict_.get('bmi'))>33) and (int(dict_.get('skin_thickness'))<21):
        return 0
    else:
        return 1

def get_r10(dict_):

    if (int(dict_.get('pregnancies'))!=0) and (int(dict_.get('pregnancies'))<4):
        return 0
    else:
        return 1

def get_r11_r12_r13_r14_r15(dict_):

    r11 = float(dict_.get('bmi')) * int(dict_.get('skin_thickness'))
    r12 = int(dict_.get('pregnancies')) / int(dict_.get('age'))
    r13 = int(dict_.get('glucose')) / dict_.get('diabetes_pedigree_function')
    r14 = int(dict_.get('age')) * dict_.get('diabetes_pedigree_function')
    
    if r11<1034:
        r15 = 0
    else:
        r15= 1
    
    return r11,r12,r13,r14,r15

def predict_pipeline(dict_):
    to_predict = []
    for i in dict_.values():
        to_predict.append(i)
    to_predict.append(get_r1(dict_))
    to_predict.append(get_r2(dict_))
    to_predict.append(get_r3(dict_))
    to_predict.append(get_r4(dict_))
    to_predict.append(get_r5(dict_))
    to_predict.append(get_r6(dict_))
    to_predict.append(get_r7(dict_))
    to_predict.append(get_r8(dict_))
    to_predict.append(get_r9(dict_))
    to_predict.append(get_r10(dict_))
    r11,r12,r13,r14,r15 = get_r11_r12_r13_r14_r15(dict_)
    to_predict.append(r11)
    to_predict.append(r12)
    to_predict.append(r13)
    to_predict.append(r14)
    to_predict.append(r15)
    
    return to_predict

    
    
    







@app.get("/")
def main():
    return {
        'message':'hello friend.'
    }

@app.post('/predict-lgbm')
def predict(infos : Infos = None):

    dict_ = dict(infos)   
    data  = predict_pipeline(dict_)
    data = np.array(data).reshape(1,-1)
    result = lgmb_model.predict(data)
    prediction = 'Diyabetsiniz' if(int(result[0]) == 1) else 'Diyabet Değilsiniz'

    return {
        'outcome':int(result[0]),
        'prediction':prediction
    }


@app.post('/predict-bagging')
def predict(infos : Infos = None):

    dict_ = dict(infos)   
    data  = predict_pipeline(dict_)
    data = np.array(data).reshape(1,-1)
    scaled_data = std_scaler.transform(data)
    result = bag_model.predict(scaled_data)
    prediction = 'Diyabetsiniz' if(int(result[0]) == 1) else 'Diyabet Değilsiniz'

    return {
        'outcome':int(result[0]),
        'prediction':prediction
    }



if __name__ == "__main__":
    uvicorn.run(app, host=FAST_API_HOST, port=FAST_API_PORT)

