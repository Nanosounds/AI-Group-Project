import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Sample data for over-the-counter drugs and associated symptoms
# Replace this with actual data
data = {
    'symptoms': [
        'headache', 'fever', 'cough', 'sore throat', 'runny nose', 
        'muscle pain', 'nausea', 'diarrhea', 'fatigue', 'chest pain'
    ],
    'drug_type': [
        'pain reliever', 'fever reducer', 'cough suppressant', 
        'throat lozenge', 'decongestant', 'muscle relaxant', 
        'anti-nausea', 'anti-diarrheal', 'energy supplement', 'pain reliever'
    ]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# We will convert categorical text into numeric features using one-hot encoding
df_encoded = pd.get_dummies(df['symptoms'])

# Labels (drug type)
labels = pd.factorize(df['drug_type'])[0]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded, labels, test_size=0.3, random_state=42)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
nn_model.fit(X_train, y_train)

# Predictions from both models
dt_predictions = dt_model.predict(X_test)
nn_predictions = nn_model.predict(X_test)

# Function to evaluate model performance
def evaluate_model(true_labels, predicted_labels, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"Precision: {precision_score(true_labels, predicted_labels, average='weighted'):.4f}")
    print(f"Recall: {recall_score(true_labels, predicted_labels, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predicted_labels, average='weighted'):.4f}")
    print("\nClassification Report:\n", classification_report(true_labels, predicted_labels))

# Evaluate Decision Tree
evaluate_model(y_test, dt_predictions, "Decision Tree")

# Evaluate Neural Network
evaluate_model(y_test, nn_predictions, "Neural Network")
