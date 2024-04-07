import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import pymongo
import html5lib

years = list(range(2022, 2020, -1))
all_matches = []
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, 'html.parser')
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text, 'html.parser')
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(1)

match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]

# Save DataFrame to CSV
match_df.to_csv("matches2.csv", index=False)

# Load CSV data and upload to MongoDB
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--uri', required=True, help="MongoDB URI with username/password")
args = parser.parse_args()

client = pymongo.MongoClient(args.uri)
db = client.matchpredictor
collection = db.matches

# Read CSV into DataFrame
match_df = pd.read_csv("matches2.csv")

# Convert DataFrame to dictionary
matches_dict = match_df.to_dict(orient='records')

# Insert data into the collection
collection.insert_many(matches_dict)

print("Data uploaded to Azure Cosmos DB MongoDB successfully.")
