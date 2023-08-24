import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("collegePlace.csv")  # Replace with your dataset file

# Preprocessing
categorical_features = ["Gender", "Stream"]
numerical_features = ["Age", "Internships", "CGPA", "Hostel", "HistoryOfBacklogs"]
target = "PlacedOrNot"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

X = data.drop(target, axis=1)
y = data[target]

# Fit the preprocessor on the entire dataset
preprocessor.fit(X)

# Save the preprocessor using pickle
with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_encoded = preprocessor.transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Save the trained model to 'model.pkl' file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
