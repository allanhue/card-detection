import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Read the dataset after having downloaded it from kaggle  
credit_detection = pd.read_csv(os.path.join('cardfile', 'creditcard.csv'))

# Basic information about the dataset
print("\nDataset Info:")
print(credit_detection.info())

# Check for missing values
print("\nMissing Values:")
print(credit_detection.isnull().sum())


# Class distributon apply data analysis 
print("\nClass Distribution:")
print(credit_detection['Class'].value_counts())
print("\nClass Distribution Percentage:")
print(credit_detection['Class'].value_counts(normalize=True) * 100)

# Create a copy of the dataset for manipulation
df = credit_detection.copy()

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features  okay not sure about this still
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier manipulation
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Basic statistics of the scaled features
print("\nBasic Statistics of Scaled Features:")
print(X_train_scaled.describe())

# Save the processed data
print("\nSaving processed data")
X_train_scaled.to_csv(os.path.join('cardfile', 'X_train_scaled.csv'), index=False)
X_test_scaled.to_csv(os.path.join('cardfile', 'X_test_scaled.csv'), index=False)
y_train.to_csv(os.path.join('cardfile', 'y_train.csv'), index=False)
y_test.to_csv(os.path.join('cardfile', 'y_test.csv'), index=False)

print("\nData processing completed successfully!")


# Initialize and train the model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Create a confusion matrix heatmap
# Confusion Matrix with Percentages
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]  # Convert to percentages

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], 
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix (Counts)', pad=20, fontsize=14)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)

# Add a second heatmap layer for percentages
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j+0.5, i+0.3, f"{cm_percent[i,j]:.1%}", 
                 ha='center', va='center', color='red')

plt.show()

# Convert classification report to DataFrame
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().drop('support', axis=1)

plt.figure(figsize=(10, 4))
sns.heatmap(report_df, annot=True, cmap='Greens', fmt='.2f', 
            cbar=False, linewidths=0.5)
plt.title('Classification Report (Precision, Recall, F1-Score)', 
          pad=15, fontsize=14)
plt.xticks(rotation=45)
plt.show()



# implement the model with seaborn and matplotlib so have a better visualisation   showing  report of the  fraud with time and amount 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Fraud Distribution Over Time (Binned)
df['Hour'] = (df['Time'] // 3600) % 24  # Convert seconds to hours
sns.histplot(data=df, x='Hour', hue='Class', bins=24, 
             palette={0: 'blue', 1: 'red'}, 
             alpha=0.6, ax=ax1)
ax1.set_title('Fraud Distribution by Hour', fontsize=14)
ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Transaction Count', fontsize=12)

# Fraud Amount Distribution (Log Scale)
sns.boxplot(data=df, x='Class', y='Amount', 
            palette={0: 'blue', 1: 'red'}, ax=ax2)
ax2.set_yscale('log')  # Handle skewed amounts
ax2.set_title('Transaction Amount by Class (Log Scale)', fontsize=14)
ax2.set_xticklabels(['Normal', 'Fraud'])
ax2.set_xlabel('Class', fontsize=12)
ax2.set_ylabel('Amount (Log Scale)', fontsize=12)

plt.tight_layout()
plt.show()



