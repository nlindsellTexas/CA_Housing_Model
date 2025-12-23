import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load data
DATA_PATH = "../data/raw/housing.csv"
df = pd.read_csv(DATA_PATH)

df["rooms_per_household"]=df["total_rooms"]/df["households"]
df["bedrooms_per_room"]=df["total_bedrooms"]/df["total_rooms"]
df["population_per_household"] = df["population"]/df["households"]


# Separate features & target
X = df.drop(columns="median_house_value")
y = df["median_house_value"]

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify column types
numeric_features = X.select_dtypes(include=["float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessing pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Full model pipeline
alphas = [float(i) for i in range(1, 21)]

for alpha in alphas:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"Alpha = {alpha:.1f} | Validation RMSE: ${rmse:,.0f}")


# Predicted vs actual
# plt.figure(figsize=(6, 6))
# plt.scatter(y_val, y_pred, alpha=0.8)
# plt.plot([y_val.min(), y_val.max()],
#          [y_val.min(), y_val.max()],
#          "r--")

# plt.xlabel("Actual House Value")
# plt.ylabel("Predicted House Value")
# plt.title("Predicted vs Actual (Validation)")
# plt.tight_layout()
# plt.grid(True)
# plt.show()

#residuals = y_val - y_pred

# plt.figure(figsize=(6, 4))
# plt.hist(residuals, bins=50)
# plt.xlabel("Residual (Actual - Predicted)")
# plt.ylabel("Count")
# plt.title("Residual Distribution")
# plt.tight_layout()
# plt.show()

