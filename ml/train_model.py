import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Load dataset

data = pd.read_csv("../dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Remove ID

data.drop("customerID", axis=1, inplace=True)


# Convert TotalCharges

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data.dropna(inplace=True)


# Split features and target

X = data.drop("Churn", axis=1)

y = data["Churn"].map({
    "Yes":1,
    "No":0
})


# Columns

categorical_columns = X.select_dtypes(
    include="object"
).columns


numeric_columns = X.select_dtypes(
    exclude="object"
).columns



# Preprocessor

preprocessor = ColumnTransformer(
    [
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        ),

        (
            "num",
            StandardScaler(),
            numeric_columns
        )
    ]
)



# Full pipeline

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ]
)



# Train

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



# Save complete model

pickle.dump(
    model,
    open("churn_model.pkl","wb")
)


print("Training completed")
print("Model saved")