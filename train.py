import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# ------------------
# Load dataset
# ------------------
df = pd.read_csv("crop_india.csv")

print("Columns:", df.columns)

# ------------------
# Clean data
# ------------------
df = df.dropna()

# Encode categorical columns
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Season"] = le_season.fit_transform(df["Season"])

# ------------------
# Features & Target
# ------------------
#features = ["State", "Crop_Year", "Season", "Area", "Production"]
features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

target = "Yield"

X = df[features]
y = df[target]

# ------------------
# Train/Test split
# ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------
# Train RandomForest
# ------------------
model = RandomForestRegressor(n_estimators=200)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Accuracy (R² score):", score)

# ------------------
# Save model
# ------------------
pickle.dump(model, open("model_real.pkl", "wb"))

print("Model saved as model_real.pkl ✅")
