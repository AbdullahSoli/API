from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the models and scalers
model_dbscan = joblib.load('dbscan_model.joblib')
scaler_dbscan = joblib.load('dbscan_scaler.joblib')

model_kmeans = joblib.load('kmeans_model.joblib')
scaler_kmeans = joblib.load('kmeans_scaler.joblib')

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    Position: str
    Age: int
    Appearances: int
    Goals: int

# Position mapping for converting to numerical values
position_mapping = {
    'Attack Centre-Forward': 1,
    'Attack LeftWinger': 2,
    'Attack RightWinger': 3,
    'Attack Second Striker': 4,
    'Defender Centre-Back': 5,
    'Defender Left-Back': 6,
    'Defender Right-Back': 7,
    'Goalkeeper': 8,
    'Midfield Attacking Midfield': 9,
    'Midfield Central Midfield': 10,
    'Midfield Defensive Midfield': 11,
    'Midfield Left Midfield': 12,
    'Midfield Right Midfield': 13
}

def preprocessing(input_features: InputFeatures):
    # Convert the position to its corresponding numerical value
    position_value = position_mapping.get(input_features.Position, 0)
    
    # Create a feature dictionary with numerical position
    dict_f = {
        'Position': position_value,
        'Age': input_features.Age, 
        'Appearances': input_features.Appearances, 
        'Goals': input_features.Goals
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in ['Position', 'Age', 'Appearances', 'Goals']]
    
    # Scale the input features
    scaled_features = scaler_kmeans.transform([features_list])
    return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model_kmeans.predict(data)
    return {"pred": y_pred.tolist()[0]}

@app.get("/")
def root():
    return "Welcome To Tuwaiq Academy"

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
