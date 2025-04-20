from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train a basic model
model_v1 = LogisticRegression()
model_v1.fit(X_train, y_train)

# Evaluate
y_pred = model_v1.predict(X_test)
acc_v1 = accuracy_score(y_test, y_pred)
print(f"Model v1 Accuracy: {acc_v1:.2f}")

model_v2 = LogisticRegression(C=0.5, max_iter=200)
model_v2.fit(X_train, y_train)
y_pred2 = model_v2.predict(X_test)
acc_v2 = accuracy_score(y_test, y_pred2)
print(f"Model v2 Accuracy: {acc_v2:.2f}")

result = {
    "model_v1":acc_v1,
    "model_v2":acc_v2
}
with open("performance.txt",'w') as f:
    for ver, acc in result.items():
        f.write(f"{ver} Accuracy : {acc}")
        
os.makedirs("models",exist_ok=True)
with open("models/model_v1.pkl",'wb') as f:
    pickle.dump(model_v1,f)

with open("models/model_v2.pkl",'wb') as f:
    pickle.dump(model_v2,f)
    
print("\t\t\t\t\t\t<----Completed---->")

