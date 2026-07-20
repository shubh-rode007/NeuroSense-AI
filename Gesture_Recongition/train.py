#train.pt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
print("Training model... (this is fast on CPU)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Training Complete! Accuracy: {acc*100:.2f}%")

# Save the brain
joblib.dump(model, 'gesture_model.pkl')
print("Model saved as 'gesture_model.pkl'")