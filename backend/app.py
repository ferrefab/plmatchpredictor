import argparse
from flask import Flask, send_file, request, jsonify
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from joblib import load
import pandas as pd
import numpy as np
import os
from flask_cors import CORS
import pymongo

app = Flask(__name__, static_url_path='', static_folder='../frontend')
CORS(app)

parser = argparse.ArgumentParser(description='Upload a file to Azure Blob Storage.')
parser.add_argument('-c', '--connection', required=True, help="Azure storage connection string")
parser.add_argument('-u', '--uri', required=True, help="Cosmos DB URI with username/password")
args = parser.parse_args()

# Verbindung zu MongoDB herstellen
database_name = "matchpredictor"
collection_name = "matches"

def load_data_from_mongodb(uri, database_name, collection_name):
    client = pymongo.MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]
    data = list(collection.find())
    return pd.DataFrame(data)

matches = load_data_from_mongodb(args.uri, database_name, collection_name)

matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype("int")

predictors = ["venue_code", "opp_code", "day_code"]

# Connect to Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(args.connection)

# Download model file from Azure Blob Container
container_name = "premierleaguepredictor3"
blob_client = blob_service_client.get_blob_client(container=container_name, blob="modelNew.pkl")
downloaded_model_path = "modelNew.pkl"
with open(downloaded_model_path, "wb") as download_file:
    download_file.write(blob_client.download_blob().readall())

# Load the model
model = load(downloaded_model_path)

# Prediction Gewinnwahrscheinlichkeit von beiden Teams wird verglichen
def predict_outcome(team1, team2):
    if team1 not in matches["team"].unique() or team2 not in matches["team"].unique():
        return "One or both teams are not found in the database."

    team1_data = matches[matches["team"] == team1]
    team2_data = matches[matches["team"] == team2]

    team1_features = team1_data.iloc[-1:][predictors]
    team2_features = team2_data.iloc[-1:][predictors]
    
    team1_pred = model.predict_proba(team1_features)[:, 1]
    team2_pred = model.predict_proba(team2_features)[:, 1]
    
    if np.mean(team1_pred) > np.mean(team2_pred):
        return f"{team1} is predicted to win!"
    elif np.mean(team1_pred) < np.mean(team2_pred):
        return f"{team2} is predicted to win!"
    else:
        return "This game will end in a draw"

# Flask routes
@app.route('/')
def index():
    return send_file("../frontend/index.html")

@app.route('/api/predict', methods=['GET'])
def predict_api():
    team1 = request.args.get('team1', default="", type=str)
    team2 = request.args.get('team2', default="", type=str)
    prediction = predict_outcome(team1, team2)
    return jsonify({
        'prediction': prediction,
        'team1': team1,
        'team2': team2
    })

if __name__ == '__main__':
    app.run(debug=True)
