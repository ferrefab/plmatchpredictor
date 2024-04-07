import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import os
from azure.storage.blob import BlobServiceClient
import joblib
import argparse
import pymongo

parser = argparse.ArgumentParser(description='Upload a file to Azure Blob Storage.')
parser.add_argument('-c', '--connection', required=True, help="Azure storage connection string")
parser.add_argument('-u', '--uri', required=True, help="Cosmos DB URI with username/password")
args = parser.parse_args()

# Define function to load data from MongoDB
def load_data_from_mongodb(uri, database_name, collection_name):
    client = pymongo.MongoClient(uri)
    db = client[database_name]
    collection = db[collection_name]
    data = list(collection.find())
    return pd.DataFrame(data)

# Process dataset
def process_dataset(matches):
    matches["date"] = pd.to_datetime(matches["date"])
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["day_code"] = matches["date"].dt.dayofweek
    matches["target"] = (matches["result"] == "W").astype("int")
    return matches

# Define function to predict outcome using the trained model
def predict_outcome(team1, team2, data, model, predictors):
    team1_data = data[data["team"] == team1]
    team2_data = data[data["team"] == team2]
    
    team1_features = team1_data.iloc[-1:][predictors]
    team2_features = team2_data.iloc[-1:][predictors]
    
    team1_pred = model.predict_proba(team1_features)[:, 1]  # Probability of winning
    team2_pred = model.predict_proba(team2_features)[:, 1]  # Probability of winning
    
    if np.mean(team1_pred) > np.mean(team2_pred):
        outcome = f"{team1} wins"
    elif np.mean(team1_pred) < np.mean(team2_pred):
        outcome = f"{team2} wins"
    else:
        outcome = "Draw"
    
    return outcome

# serialize model and save as binary file
def serialize_model(model, file_path):
    joblib.dump(model, file_path)

def upload_to_blob(args, model):
    base_container_name = 'premierleaguepredictor' 
    local_file_name = 'modelNew.pkl' 
    blob_name = os.path.basename(local_file_name)

    try:
        print("Starting file upload to Azure Blob Storage...")

        blob_service_client = BlobServiceClient.from_connection_string(args.connection)

        existing_containers = blob_service_client.list_containers(name_starts_with=base_container_name)
        next_suffix = len(list(existing_containers)) + 1  # Nummerierung für den nächsten Container

        container_name = f"{base_container_name}{next_suffix}"

        if not any(container['name'] == container_name for container in existing_containers):
            blob_service_client.create_container(container_name)
            print(f"Container '{container_name}' created.")

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        serialize_model(model, local_file_name)

        with open(local_file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"File '{local_file_name}' uploaded to container '{container_name}' as blob '{blob_name}'.")

    except Exception as ex:
        print('Error:', ex)

# Load match data from MongoDB
def main(args):
    database_name = "matchpredictor"
    collection_name = "matches"
    matches = load_data_from_mongodb(args.uri, database_name, collection_name)

    # Process dataset
    matches = process_dataset(matches)

    # Define predictors
    predictors = ["venue_code", "opp_code", "day_code"]

    # Initialize and train
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    train = matches[matches["date"] < '2024-01-01']
    test = matches[matches["date"] >= '2024-01-01']
    rf.fit(train[predictors], train["target"])

    # Make predictions
    preds = rf.predict(test[predictors])

    # Evaluate predictions
    accuracy = accuracy_score(test["target"], preds)
    precision = precision_score(test["target"], preds)
    print(f"Accuracy: {accuracy}, Precision: {precision}")

    # TEST
    team1 = "West Ham United"
    team2 = "Manchester United"
    predicted_outcome = predict_outcome(team1, team2, matches, rf, predictors)
    print(predicted_outcome)

    upload_to_blob(args, rf)

if __name__ == "__main__":
    main(args)
