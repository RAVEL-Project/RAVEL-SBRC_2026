import torch
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile  , join
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def data_in(arquivo):
    features=pd.DataFrame()
    labels=pd.DataFrame()

    if  "feature" in arquivo:
        features=pd.read_pickle(arquivo)

    if "label" in arquivo:
        labels=pd.read_pickle(arquivo)["label"]
            
    return features, labels 


def exp_pred(X_test, y_test, model, encoding_time, training_time):

    # ==================================================
    # Prediction
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_prediction = time.perf_counter()

    with torch.no_grad():
        predictions = model.predict(X_test)

    if device.type == "cuda":
        torch.cuda.synchronize()
    prediction_time = time.perf_counter() - start_prediction
     # ==================================================
    # Metrics
    # ==================================================
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")
    conf_matrix = confusion_matrix(y_test, predictions)

    # Save
    results.append({
        "bits_encoding": bits_encoding,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "encoding_time_sec": encoding_time,
        "training_time_sec": training_time,
        "prediction_time_sec": prediction_time,
        "confusion_matrix": conf_matrix
    })

    return results,prediction_time,acc,precision,recall,f1,conf_matrix

   

def exp_train(X_train,  y_train):


    if device.type == "cuda":
        torch.cuda.synchronize()
    

    # ==================================================
    # Model
    # ==================================================
    entry_size = X_train.shape[1]
    # model = Wisard(entry_size, num_classes, tuple_size=8)
    model = BloomWisard(entry_size, num_classes, tuple_size=8, capacity=100, error=0.8)
    # ==================================================
    # Training
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_training = time.perf_counter()

    with torch.no_grad():
        model.fit(X_train, y_train)

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start_training
    return training_time, model

def encoding(X_train_raw,X_test_raw):

    # ==================================================
    # Encoding
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_encoding = time.perf_counter()

    encoding = Thermometer(bits_encoding).fit(X_train_raw)
    X_train = encoding.binarize(X_train_raw).flatten(start_dim=1)
    X_test = encoding.binarize(X_test_raw).flatten(start_dim=1)
    encoding_time = time.perf_counter() - start_encoding
    return X_train, X_test,encoding_time



#git hub e python setup.py install
from torchwnn.datasets.iris import Iris
#from sklearn.datasets import load_iris
from torchwnn.classifiers import BloomWisard
from torchwnn.encoding import Thermometer

# ==================================================
# Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ==================================================
# Load dataset
# ==================================================

iris=Iris()
features=iris.features
labels=iris.labels
num_classes=iris.num_classes


# features=pd.DataFrame()
# labels=pd.DataFrame()
# pasta_csv='data/'
# for arquivo in sorted(listdir(pasta_csv),key=lambda s: (len(s), s)):
#     file_path=join(pasta_csv, arquivo)
#     if isfile(file_path) and ".pkl" in arquivo:
#         f, l=data_in(file_path)
#         features = pd.concat([features, f], ignore_index=True)
#         labels = pd.concat([labels, l], ignore_index=True)



# num_classes=pd.read_pickle('./data/n_classes.pkl').squeeze()

X = torch.tensor(features.values).to(device)
y = torch.tensor(list(labels)).squeeze().to(device)

# ==================================================
# FIXED train / test split
# ==================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# ==================================================
# Values to test
# ==================================================
bits_values = [5]#, 10, 15, 20, 25, 30]

results = []

for bits_encoding in bits_values:
    print(f"\nRunning experiment with bits_encoding = {bits_encoding}")
    X_train, X_test,encoding_time=encoding(X_train_raw,X_test_raw)
    
    training_time, model=exp_train(X_train, y_train)
    
    results,prediction_time,acc,precision,recall,f1,conf_matrix=exp_pred(X_test, y_test, model, encoding_time, training_time)

    # Print summary
    print(
        f"Bits = {bits_encoding:2d} | "
        f"Acc = {acc:.4f} | "
        f"Prec = {precision:.4f} | "
        f"Recall = {recall:.4f} | "
        f"F1 = {f1:.4f}"
    )
    print("Confusion matrix:")
    print(conf_matrix)
    print (f"\nEncoding time {encoding_time} seconds")
    print (f"Train time {training_time} seconds")
    print (f"Prediction time {training_time} seconds")
