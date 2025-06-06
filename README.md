# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using the Kaggle Credit Card Fraud Detection dataset.

## Project Overview
This project implements a machine learning model to detect fraudulent credit card transactions. It includes:
- Data preprocessing and handling class imbalance
- Feature scaling and engineering
- Logistic Regression model implementation
- Comprehensive model evaluation and visualization
- Time-based and amount-based fraud analysis

## Dataset
The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. Due to GitHub's file size limitations, the dataset is not included in this repository.

### How to Get the Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place the `creditcard.csv` file in the `cardfile` directory
3. The dataset will be automatically ignored by Git (as specified in `.gitignore`)

## Setup and Installation
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file with your Kaggle credentials:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_key
   ```

## Project Structure
- `mainpy/`: Contains the main Python scripts
- `cardfile/`: Directory for storing the dataset
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked by Git)

## Features
- Data preprocessing and scaling
- Class imbalance handling
- Model training and evaluation
- Visualization of fraud patterns
- Time-based analysis of fraudulent transactions
- Amount distribution analysis

## Results
The project includes various visualizations:
- Confusion matrix
- Classification report
- Fraud distribution by hour
- Transaction amount analysis

## Contributing
Feel free to submit issues and enhancement requests!

## Tech Stack
- Python 3.11
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
- Git for version control

## Troubleshooting Procedures
- **Large File Issue**: The dataset file `cardfile/creditcard.csv` exceeded GitHub's file size limit (100 MB). To resolve this, I used `git filter-repo` to remove the file from the entire Git history and force-pushed the cleaned history to GitHub.
- **Git History Cleanup**: After removing the large file, I re-added the origin remote and force-pushed the changes to ensure the repository was updated correctly.

## Problems Encountered
- **GitHub File Size Limit**: The initial push was rejected due to the large dataset file. This was resolved by cleaning the Git history.
- **Git Remote Removal**: The `git filter-repo` command removed the origin remote, which required re-adding it before pushing changes.


<!-- Summary Checklist
 Load and explore data

 Preprocess and handle imbalance

 Split into train/test sets

 Build and train a model

 Evaluate with proper metrics

 Visualize results

 Iterate and improve -->

