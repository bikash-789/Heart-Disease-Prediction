# 1. Import Libraries
import uvicorn
from fastapi import FastAPI
from Patient import Patient
import pickle

# 2. Create the app object
app = FastAPI()
classifier = pickle.load(open('./HeartDisease_Model.pkl', 'rb'))

# 3. Index route, open automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello World!'}

# 4. Route with a single parameter, returns the parameter within a message
# Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello {name}'}

# 5. Expose the prediction functionality, make a prediction from the passed data
# JSON data and return the predicted Heart Disease Classification 
@app.post('/predict')
def predict_heartDisease(data:Patient):
    data = data.dict()
    print(data)
    age=data['age']
    sex=data['sex']
    cp=data['cp']
    trestbps=data['trestbps']
    chol=data['chol']
    fbs=data['fbs']
    restecg=data['restecg']
    thalach=data['thalach']
    exang=data['exang']
    oldpeak=data['oldpeak']
    slope=data['slope']
    ca=data['ca']
    thal=data['thal']
    print("Predicting.....")
    prediction = classifier.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if(prediction[0] == 1):
        prediction="Patient have heart disease!"
    else:
        prediction="Patient doesn't have heart disease!"
    return {
        'prediction': prediction
    }


# 5. Run the api with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload