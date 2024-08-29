from fastapi import FastAPI
app = FastAPI()
@app.get("/items/{item_id}")
async def read_item(item_id):
 return {"item_id": item_id}


import joblib
model_dbscan = joblib.load('dbscan_model.joblib')
scaler_dbscan = joblib.load('dbscan_scaler.joblib')

model_kmeans = joblib.load('kmens_model.joblib')
scaler_kmeans = joblib.load('kmens_scaler.joblib')

from pydantic import BaseModel
 # Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Position: str
    Age: int
    Appearances: int
    Goals: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
            'Position': input_features.Position == 'Defender Centre-Back',
            'Age': input_features.Age, 
            'Appearances': input_features.Appearances, 
            'Goals': input_features.Goals ,
            
        }
    return dict_f

@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)


def preprocessing(input_features: InputFeatures):
    dict_f = {
            'Position': input_features.Position == 'Defender Centre-Back',
            'Age': input_features.Age, 
            'Appearances': input_features.Appearances, 
            'Goals': input_features.Goals ,
        }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler_kmeans.transform([list(dict_f.values
 ())])
    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model_kmeans.predict(data)
    return {"pred": y_pred.tolist()[0]}
