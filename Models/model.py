import pickle

with open("../Models/model.pkl", "rb") as f:
    classifier = pickle.load(f)
