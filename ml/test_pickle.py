import pickle

files = [
"logistic_model.pkl",
"encoder.pkl",
"scaler.pkl",
"feature_columns.pkl"
]


for file in files:
    try:
        obj = pickle.load(open(file,"rb"))
        print(file, "OK")
        print(type(obj))

    except Exception as e:
        print(file, "ERROR")
        print(e)