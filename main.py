from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load models and scalers
model_kmeans = joblib.load('kmens_model.joblib')
scaler_kmeans = joblib.load('kmens_scaler.joblib')

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Position: str
    Age: int
    Appearances: int
    Goals: int

# Data preprocessing function
def preprocessing(input_features: InputFeatures):
    # Convert input features to dictionary
    dict_f = {
        'Position': input_features.Position == 'Defender Centre-Back',
        'Age': input_features.Age, 
        'Appearances': input_features.Appearances, 
        'Goals': input_features.Goals
    }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler_kmeans.transform([features_list])
    return scaled_features
@app.get("/")
async def root():
    return {"message": "Welcome to the Football Player Prediction API"}
# Define prediction route
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model_kmeans.predict(data)
    return {"pred": y_pred.tolist()[0]}

# Define a test route
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
