import pandas as pd
import requests
import os
from sklearn.metrics import accuracy_score

print('Test process has started.')
PATH = os.getcwd()

test = pd.read_csv(PATH+'/data/test.csv')
outcomes = pd.read_csv(PATH+'/data/test_outcomes.csv')
test_outcomes = outcomes['Outcome']
entries = test.drop([f'R{i}' for i in range(1,16)],axis=1).rename(columns={'Pregnancies':'pregnancies','Glucose':'glucose','BloodPressure':'blood_pressure','SkinThickness':'skin_thickness','Insulin':'insulin','BMI':'bmi','DiabetesPedigreeFunction':'diabetes_pedigree_function','Age':'age'}).to_dict('records')

# test = pd.read_csv(PATH + '/data/zero_cleaned_test_data.csv')
# test_outcomes = test['Outcome']
# entries = test.drop('Outcome',axis=1).rename(columns={'Pregnancies':'pregnancies','Glucose':'glucose','BloodPressure':'blood_pressure','SkinThickness':'skin_thickness','Insulin':'insulin','BMI':'bmi','DiabetesPedigreeFunction':'diabetes_pedigree_function','Age':'age'}).to_dict('records')


# bag_predicted_outcomes = []
# print('Predicting BaggingClassifier accuracy score... \n\n')
# for entry in entries:
#     response = requests.post('http://localhost:8000/predict-bagging',json=entry)
#     bag_predicted_outcomes.append(response.json()['outcome'])

# print(f'Bagging accuracy score : {accuracy_score(test_outcomes,bag_predicted_outcomes)}')

print('Predicting LGBM accuracy score...\n')
lgbm_predicted_outcomes = []
for entry in entries:
    response = requests.post('http://localhost:8000/predict-lgbm',json=entry)
    lgbm_predicted_outcomes.append(response.json()['outcome'])

print(f'LGBM accuracy score : {accuracy_score(test_outcomes,lgbm_predicted_outcomes)}')