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
        features=pd.read_pickle(arquivo)[["src_ip","src_port","dst_ip","dst_port","duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes"]]

    if "label" in arquivo:
        labels=pd.read_pickle(arquivo)["label"]
            
    return features, labels 


def exp_pred(X_test, y_test, model, encoding_time, training_time, prediction_time):

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
    prediction_time += time.perf_counter() - start_prediction
    #  # ==================================================
    # # Metrics
    # # ==================================================
    # acc = accuracy_score(y_test, predictions)
    # precision = precision_score(y_test, predictions, average="macro")
    # recall = recall_score(y_test, predictions, average="macro")
    # f1 = f1_score(y_test, predictions, average="macro")
    # conf_matrix = confusion_matrix(y_test, predictions)

    # # Save
    results.append({
    #     "bits_encoding": bits_encoding,
    #     "accuracy": acc,
    #     "precision_macro": precision,
    #     "recall_macro": recall,
    #     "f1_macro": f1,
         "encoding_time_sec": encoding_time,
         "training_time_sec": training_time,
         "prediction_time_sec": prediction_time,
    #     "confusion_matrix": conf_matrix
    })

    # return results,prediction_time,acc,precision,recall,f1,conf_matrix
    return results,prediction_time

   

def exp_train(X_train,  y_train,training_time,model):


    if device.type == "cuda":
        torch.cuda.synchronize()
    



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
    training_time += time.perf_counter() - start_training
    return training_time, model

def encoding(X_train_raw,X_test_raw,encoding_time,bits_encoding):

    # ==================================================
    # Encoding
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_encoding = time.perf_counter()

    encoding = Thermometer(bits_encoding).fit(X_train_raw)
    X_train = encoding.binarize(X_train_raw).flatten(start_dim=1)
    X_test = encoding.binarize(X_test_raw).flatten(start_dim=1)
    encoding_time += time.perf_counter() - start_encoding
    return X_train, X_test,encoding_time



#git hub e python setup.py install
#from torchwnn.datasets.iris import Iris
#from sklearn.datasets import load_iris
from torchwnn.classifiers import Wisard
from torchwnn.encoding import Thermometer

# ==================================================
# Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")



#X = torch.tensor(features.values).to(device)
#y = torch.tensor(list(labels.squeeze())).to(device)


#features, labels =None,None
# ==================================================
# FIXED train / test split
# ==================================================
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(
#     X, y, test_size=0.75, random_state=0, stratify=y
# )
# X,y=None,None
# ==================================================
# Values to test
# ==================================================
# bits_values = [15]#5, 10, 15, 20, 25, 30]


# for bits_encoding in bits_values:
#     print(f"\nRunning experiment with bits_encoding = {bits_encoding}")
#     X_train, X_test,encoding_time=encoding(X_train_raw,X_test_raw,encoding_time)
#     X_train_raw,X_test_raw =None,None
#     print(f"\nEND OF ENCODING")
#     print(f"\nSTART OF TRAIN")    
#     training_time, model=exp_train(X_train, y_train,training_time)
#     print(f"\nEnd  OF TRAIN")
#     print(f"\nSTART OF little TRAIN")
#     #results,prediction_time,acc,precision,recall,f1,conf_matrix=exp_pred(X_test, y_test, model, encoding_time, training_time,prediction_time)
#     results,prediction_time=exp_pred(X_test, y_test, model, encoding_time, training_time,prediction_time)
#     print(f"\nEND OF little TRAIN")

def main(X_train_raw,X_test_raw,encoding_time,training_time,prediction_time,model,results,bit_encoding):
    #print(f"\nRunning experiment with bits_encoding = {bits_encoding}")
    X_train, X_test,encoding_time=encoding(X_train_raw,X_test_raw,encoding_time,bit_encoding)
    X_train_raw,X_test_raw =None,None
    #print(f"\nEND OF ENCODING")
    #print(f"\nSTART OF TRAIN")    
    training_time, model=exp_train(X_train, y_train,training_time,model)
    #print(f"\nEnd  OF TRAIN")
    #print(f"\nSTART OF little TRAIN")
    #results,prediction_time,acc,precision,recall,f1,conf_matrix=exp_pred(X_test, y_test, model, encoding_time, training_time,prediction_time)
    results,prediction_time=exp_pred(X_test, y_test, model, encoding_time, training_time,prediction_time)
    #print(f"\nEND OF little TRAIN")
    return encoding_time,training_time,prediction_time,model,results

results = []
encoding_time=0
prediction_time=0
training_time=0
# ==================================================
# Load dataset
# ==================================================

# iris=Iris()
# features=iris.features
# labels=iris.labels
#num_classes=iris.num_classes
features=pd.DataFrame()
labels=pd.DataFrame()
pasta_csv='data/'
for arquivo in sorted(listdir(pasta_csv),key=lambda s: (len(s), s)):
    file_path=join(pasta_csv, arquivo)
    print(file_path+'                            ', end="\r")
    if isfile(file_path) and ".pkl" in arquivo:
        f, l=data_in(file_path)
        features = pd.concat([features, f], ignore_index=True)
        labels = pd.concat([labels, l], ignore_index=True)
        #break
f,l=None,None

labels=labels.astype('int')


num_classes=pd.read_pickle('./data/n_classes.pkl').squeeze()
# ==================================================
# Model
# ==================================================
print(' start training                       ', end="\r")
entry_size = features.values.shape[1]
model = Wisard(entry_size, num_classes, tuple_size=8)

#batch implementation
batch_size=256
l_data=len(features.index)
bit_t=10
er=0
for i in range(0, l_data, batch_size):
    if l_data >=i+batch_size: 

        X = torch.tensor(features[i:i+batch_size].values).to(device)
        y = torch.tensor(list(labels[i:i+batch_size].squeeze())).to(device)
        #print(len(y),len(X), end="\r")
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.75, random_state=0, stratify=y
            )
            X,y=None,None        
        except:
            er+=1
           
            continue

        encoding_time,training_time,prediction_time,model,results = main(X_train_raw,X_test_raw,encoding_time,training_time,prediction_time,model,results,bit_t)
    else: 
        X = torch.tensor(features[i:].values).to(device)
        y = torch.tensor(list(labels[i:].squeeze())).to(device)
        #print(f'{len(y)/l_data}% do dataset processado', end="\r")

        encoding_time,training_time,prediction_time,model,results= main(X_train_raw,X_test_raw,encoding_time,training_time,prediction_time,model,results,bit_t)

# Print summary
# print(
#     f"Bits = {bits_encoding:2d} | "
#     f"Acc = {acc:.4f} | "
#     f"Prec = {precision:.4f} | "
#     f"Recall = {recall:.4f} | "
#     f"F1 = {f1:.4f}"
# )
# print("Confusion matrix:")
# print(conf_matrix)
d=l_data/batch_size
print (f"\nEncoding time {encoding_time} seconds")
print (f"Train time {training_time} seconds")
print (f"Prediction time {prediction_time} seconds")
print(f"\n {er} | {er/d} de batchs perdidos.")
