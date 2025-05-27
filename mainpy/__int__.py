import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Load environment variables
load_dotenv()

# Get credentials from environment variables
username = os.getenv('KAGGLE_USERNAME')
key = os.getenv('KAGGLE_KEY')

# Configure Kaggle API using environment variables
api = KaggleApi()
api.authenticate()


# Read the dataset directly from Kaggle
dataset = 'mlg-ulb/creditcardfraud'
file_name = 'creditcard.csv'

# Read the dataset directly into a pandas DataFrame
credit_detection = pd.read_csv(f'kaggle://{dataset}/{file_name}')

# Display the first few rows to verify
print("Dataset loaded successfully!")
credit_detection.head()

