import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Dataset load karein (IRIS.csv aapke isi folder me hona chahiye)
try:
    df = pd.read_csv('IRIS.csv')
    print("Dataset successfully load ho gaya hai!")
except FileNotFoundError:
    print("Error: IRIS.csv file is folder me nahi mili.")
    exit()

# 2. X (Features) aur y (Target/Species) ko alag karein
X = df.drop(columns=['species'])
y = df['species']

# 3. Data ko Train aur Test sets me split karein (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Machine Learning Model (Random Forest) ko train karein
print("Model ki training shuru ho rahi hai...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model ko test karein aur accuracy check karein
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Trained model ko save karein taaki baad me use ho sake
joblib.dump(model, 'iris_model.pkl')
print("\nModel successfully save ho gaya: 'iris_model.pkl'")

