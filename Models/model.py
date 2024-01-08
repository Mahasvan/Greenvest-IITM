import pickle

with open("model.plk", "rb") as f:
    classifier = pickle.load(f)
