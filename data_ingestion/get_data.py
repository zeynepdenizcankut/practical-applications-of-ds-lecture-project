import requests
import pandas as pd
from datetime import datetime, timedelta
import time

pd.set_option('display.max_columns', None)

# 1. Configuration
API_KEY = "cvmNVUH4Db7ea9Bj2Al5LFaud1qMyXK3Hc0eIDdA"  
YEARS_BACK = 5
TOTAL_RECORDS_TO_FETCH = 30000 

# 2. Calculate Date Range
end_date = datetime.now()
start_date = end_date - timedelta(days=YEARS_BACK * 365)
start_str = start_date.strftime('%Y%m%d')
end_str = end_date.strftime('%Y%m%d')

# 3. Construct Search Query 
drugs = [
    'SEMAGLUTIDE', 'OZEMPIC', 'WEGOVY', 
    'TIRZEPATIDE', 'MOUNJARO', 'ZEPBOUND'
]
drug_string = " OR ".join([f'"{d}"' for d in drugs])
search_query = f'patient.drug.medicinalproduct:({drug_string}) AND receivedate:[{start_str} TO {end_str}]'

base_url = "https://api.fda.gov/drug/event.json"
all_results = []

print(f"Searching for: {drug_string}")

# 4. Fetch Data
start_time = time.time()
for skip in range(0, TOTAL_RECORDS_TO_FETCH, 1000):
    params = {
        "api_key": API_KEY,
        "search": search_query,
        "limit": 1000,
        "skip": skip
    }
    
    print(f"Fetching records {skip} to {skip + 1000}...")
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if not results:
                print("No more results found.")
                break
            all_results.extend(results)
            print(f"Added {len(results)} records.")
        elif response.status_code == 429:
            print("Rate limit hit! Waiting 30 seconds...")
            time.sleep(30)
        else:
            print(f"Server Error {response.status_code}: {response.text}")
            break
            
    except Exception as e:
        print(f"Connection Error: {e}")
        break
    
    time.sleep(1.0)

# 5. Process and Save
if all_results:
    print(f"\nTotal records collected: {len(all_results)}")
    df = pd.json_normalize(all_results)
    
    # Save to CSV
    output_file = "glp1_data_final.csv"
    df.to_csv(output_file, index=False)
    print(f"Successfully saved to {output_file}")
    print(f"Total Columns: {len(df.columns)}")
else:
    print("Could not find any data. Please check if your API key is active.")
end_time = time.time()
print(f"Data fetching completed in {end_time - start_time:.2f} seconds.")

df.head()
df.columns



