import os
import tempfile
import pandas as pd
from supabase import create_client
from airflow.models import Variable

def fetch_training_data(**kwargs):
    """Fetch typo correction data from Supabase"""
    # Get Supabase credentials from Airflow Variables
    supabase_url = Variable.get("SUPABASE_URL")
    supabase_key = Variable.get("SUPABASE_ANON_KEY")
    
    try:
        # Initialize Supabase client
        client = create_client(supabase_url, supabase_key)
        
        # Fetch all records from typo_training_dataset table
        response = client.table("typo_training_dataset").select("typo", "corrected", "domain").execute()
        
        if not response.data:
            print("No training data found in Supabase table")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        
        # Rename columns to match training code
        df = df.rename(columns={"typo": "query", "corrected": "correction"})
        
        print(f"Successfully fetched {len(df)} training examples from Supabase")
        
        # Save DataFrame to a temporary CSV file for passing between tasks
        temp_csv = os.path.join(tempfile.gettempdir(), "training_data.csv")
        df.to_csv(temp_csv, index=False)
        
        # Pass the CSV path via XCom
        kwargs['ti'].xcom_push(key='training_data_path', value=temp_csv)
        
        return temp_csv
        
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        raise