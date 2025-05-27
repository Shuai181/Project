# Project
Data Sources
Code:

python
import pandas as pd
df_train = pd.read_csv('Credit Card Transactions Fraud Detection Dataset/fraudTrain.csv')
df_test = pd.read_csv('Credit Card Transactions Fraud Detection Dataset/fraudTest.csv')
df = pd.concat([df_train, df_test], ignore_index=True)
Explanation:
This code loads the training and testing datasets (fraudTrain.csv and fraudTest.csv) and merges them into a unified DataFrame (df). It corresponds to the Data Sources section, as it specifies the origin of the data and initial data ingestion steps.

Methodology
Data Preprocessing:

Code:

python
# Convert datetime columns
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Clean data (drop redundant columns)
def clean_data(clean):
    clean.drop(["Unnamed: 0", 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time'], axis=1, inplace=True)
    return clean
clean_data(df)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in ['merchant', 'category', 'job', 'gender']:
    df[column] = label_encoder.fit_transform(df[column])
Explanation:

Date Conversion: Converts trans_date_trans_time and dob to datetime format to enable time-based analysis.

Feature Removal: Drops irrelevant columns (e.g., user PII, transaction IDs) to reduce noise.

Categorical Encoding: Encodes categorical variables (merchant, category, etc.) into numerical values for model compatibility.
This aligns with the Methodology section under data preparation steps.

Exploratory Data Analysis (EDA):

Code:

python
# Gender distribution (pie chart)
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')

# Transaction hour analysis (bar charts)
df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
hourly_transactions = df.groupby('transaction_hour')['cc_num'].count()
plt.bar(hourly_transactions.index, hourly_transactions.values)

# Fraud vs. non-fraud distribution
fraud_counts = df['is_fraud'].value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)

# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
Explanation:

Visualizations explore data patterns, such as gender balance, transaction frequency by hour, and class imbalance (fraud vs. non-fraud).

The heatmap identifies feature correlations (e.g., amt vs. is_fraud).
These steps are part of the Methodology section under exploratory analysis.

Model Training:

Code:

python
# Split data
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

svm_model = SGDClassifier(loss='hinge')
svm_model.fit(X_train, y_train)
Explanation:

Splits data into training/testing sets.

Trains Logistic Regression and SVM models.
This corresponds to the Methodology section under model implementation.

Results
Code:

python
# Evaluate model performance
def print_performance_metrics(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")

for model_name, model in models.items():
    print_performance_metrics(model, X_test, y_test, model_name)
Explanation:

Computes metrics (accuracy, precision, recall, F1-score) to evaluate model performance.

Highlights challenges like low recall due to class imbalance.
This aligns with the Results section, summarizing model outcomes and limitations.

Next Steps
Code Implications:
While no explicit code is written for future steps, the analysis reveals:

python
# Example: Severe class imbalance (fraud vs. non-fraud)
fraud_counts = df['is_fraud'].value_counts()
Explanation:

The extreme class imbalance (fraud: 0.6%, non-fraud: 99.4%) suggests the need for techniques like SMOTE (oversampling) or using models like XGBoost.

This informs the Next Steps section, recommending advanced methods to address imbalance and improve recall.

Conclusion
Code Insights:

python
# Fraud transaction patterns by hour
hourly_fraud_transactions = df[df['is_fraud'] == 1].groupby('transaction_hour')['cc_num'].count()
Explanation:

Fraudulent transactions peak at specific hours (e.g., midnight).

Recommends monitoring high-risk timeframes and refining feature engineering.
This supports the Conclusion section, summarizing key findings and actionable insights.

Bibliography
Code Tools:
Libraries like pandas, scikit-learn, and matplotlib are used.
