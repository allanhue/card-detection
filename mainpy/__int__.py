import numpy as np
import pandas as pd

#use this dataset 
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# Load the dataset
credit_detection=pd.read_csv(path + "/creditcard.csv")

#display the rows to see if its working
credit_detection.head()
