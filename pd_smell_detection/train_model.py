import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# reproducibility
np.random.seed(42)

# generate synthetic dataset
n_samples = 500

age = np.random.randint(30,80,n_samples)
gender = np.random.randint(0,2,n_samples)
smell_score = np.random.randint(0,21,n_samples)

# rule-based target generation
parkinson = []

for s in smell_score:
    if s <= 8:
        parkinson.append(1)
    else:
        parkinson.append(0)

data = pd.DataFrame({
    "age":age,
    "gender":gender,
    "smell_score":smell_score,
    "parkinson":parkinson
})

X = data[["age","gender","smell_score"]]
y = data["parkinson"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)

print("Model Accuracy:",accuracy)

# save model
pickle.dump(model,open("smell_pd_model.pkl","wb"))

print("Model saved as smell_pd_model.pkl")