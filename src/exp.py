
import pandas as pd
import pickle as pkl
from sklearn.metrics import (
     accuracy_score,
     precision_score,
     recall_score,
     f1_score,
     confusion_matrix
 )
from sklearn.model_selection import train_test_split
import sys
import time
import torch
from torchwnn.classifiers import Wisard, BloomWisard
from torchwnn.encoding import Thermometer

from datetime import datetime
from os import listdir, makedirs
from os.path import isfile  , join , isdir

import argparse

def data_in(arquivo):
    features=pd.DataFrame()
    labels=pd.DataFrame()

    if  "feature" in arquivo:
        features=pd.read_pickle(arquivo)[["src_ip","src_port","dst_ip","dst_port","duration","src_bytes","dst_bytes","missed_bytes","src_pkts","src_ip_bytes","dst_pkts","dst_ip_bytes"]]
        return features
    if "label" in arquivo:
        labels=pd.read_pickle(arquivo)["label"]
        return labels    
    #return features, labels 


def exp_pred(X_test,y_test, model, prediction_time,bits_encoding):

    if device.type == "cuda":
        torch.cuda.synchronize()
    X=torch.tensor(X_test.values).to(device)
    
    X,t=encoding(X,prediction_time,bits_encoding)
        
    start_prediction = time.perf_counter()
    with torch.no_grad():
        predictions = model.predict(X)

    prediction_time += time.perf_counter() - start_prediction
    
    
    return pd.DataFrame({"predictions":predictions, "labels":y_test}),prediction_time+t


   

def exp_train(X_train,  y_train,training_time,model):


    # ==================================================
    # Training
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_training = time.perf_counter()

    with torch.no_grad():
        model.fit(X_train, y_train)

    training_time += time.perf_counter() - start_training
    return training_time, model

def encoding(X_raw,encoding_time,bits_encoding):

    # ==================================================
    # Encoding
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_encoding = time.perf_counter()

    encoding = Thermometer(bits_encoding).fit(X_raw)
    X = encoding.binarize(X_raw).flatten(start_dim=1)
    
    encoding_time += time.perf_counter() - start_encoding
    return X ,encoding_time

# ==================================================
# Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def main(X_train_raw,y_train,encoding_time,training_time,model,bit_encoding):

    X_train,encoding_time=encoding(X_train_raw,encoding_time,bit_encoding)
    #X_train_raw =None

    training_time, model=exp_train(X_train, y_train,training_time,model)

    return encoding_time,training_time,model

#==================================================
# setting up 
#==================================================
parser = argparse.ArgumentParser(description="Exemplo simples com argparse")

parser.add_argument("--data", help="data path",default='./data/')
parser.add_argument("-o","--output", help="output path",default='./results/')
parser.add_argument("--model", help="bloom ou wis",default='wis')
parser.add_argument("-t","--thermometer-size", type=int, help="numero de bits do termometro",default=5)
parser.add_argument("-b","--batch-size", type=int, help="batch size",default=128)
parser.add_argument("--test_size", type=float, help="test data %",default=0.1)

# Wisard
parser.add_argument("--tuple-size", type=int, help="tuple size parameter",default=8)
# Flag (True/False)
parser.add_argument("--bleaching", action="store_true", help="set bleaching parameter true")

#BloomWisard
parser.add_argument("-c","--capacity", type=int, help="capacity parameter",default=100)
parser.add_argument("-e","--error", type=int, help="error parameter",default=0.8)

args = parser.parse_args()

print(f"config do exp {args}")
#==================================================
# setting up 
#==================================================

results = []
encoding_time=0
prediction_time=0
training_time=0
# ==================================================
# Loading dataset
# ==================================================

features=pd.DataFrame()
labels=pd.DataFrame()

pasta_csv=args.data
for arquivo in sorted(listdir(pasta_csv),key=lambda s: (len(s), s)):
    file_path=join(pasta_csv, arquivo)
    print(file_path+'                            ', end="\r")
    if isfile(file_path) and ".pkl" in arquivo:
        f=data_in(file_path)
        if "label" in arquivo:
           labels = pd.concat([labels, f], ignore_index=True)
        if "feature" in arquivo:
           features = pd.concat([features, f], ignore_index=True)
        
        
f=None
# features =data_in('data/feature_2.pkl')
# labels = data_in('data/label_2.pkl')

labels=labels.astype('int')

num_classes=pd.read_pickle('./data/n_classes.pkl').squeeze()
#==================================================
#End of Loading dataset
#==================================================


# ==================================================
# Model
# ==================================================
print(' start training                       ', end="\r")
entry_size = features.values.shape[1]
if args.model == 'wis':
    model = Wisard(entry_size, num_classes, args.tuple_size,bleaching=args.bleaching)
if args.model == 'bloom':
    model = BloomWisard(entry_size, num_classes, tuple_size=args.tuple_size, capacity=args.capacity, error=args.error) 
# ==================================================
# End Model
# ==================================================



batch_size=args.batch_size
l_data=len(features.index)

#thermometer size
bit_t=args.thermometer_size

test_features=pd.DataFrame()
test_labels=pd.Series()

er=0
for i in range(0, l_data, batch_size):
    if l_data >=i+batch_size: 
        # ==================================================
        # Iteranting dataset through mini batch
        # The mini batch is splited into train and test data
        # ==================================================
        
        #X = torch.tensor(features[i:i+batch_size].values).to(device)
        #y = torch.tensor(list(labels[i:i+batch_size].squeeze())).to(device)
        X = features[i:i+batch_size]#.values
        y = labels[i:i+batch_size]
        try:

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=0, stratify=y
            )
            X,y=None,None

            test_features=pd.concat([test_features,X_test_raw], ignore_index=True)
            test_labels=pd.concat([test_labels,y_test], ignore_index=True)
        except:
            er+=1
            continue


    else: 
        # X = torch.tensor(features[i:].values).to(device)
        # y = torch.tensor(list(labels[i:].squeeze())).to(device)
        X = features[i:]
        y = labels[i:]
        #print(f'{len(y)/l_data}% do dataset processado', end="\r")
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=0, stratify=y
            )
            X,y=None,None

            test_features=pd.concat([test_features,X_test_raw], ignore_index=True)
            test_labels=pd.concat([test_labels,y_test], ignore_index=True)
        except:
            er+=1
            continue
        
    X_train_raw=torch.tensor(X_train_raw.values).to(device)
    y_train=torch.tensor(list(y_train.squeeze())).to(device)
    encoding_time,training_time,model = main(X_train_raw,y_train,encoding_time,training_time,model,bit_t)
    X_train_raw,y_train=None,None

print(' start testing                       ', end="\r")
results,prediction_time=exp_pred(test_features,test_labels, model, prediction_time, bit_t)


#==================================================================
# Writing results
#==================================================================
d=l_data/batch_size
model_size=sys.getsizeof(model)
print (f'mAP {precision_score(list(results['predictions'].squeeze()), list(results['labels'].squeeze()))}')
print (f'Model size {model_size} bytes.')
print (f"\nEncoding time {encoding_time} seconds")
print (f"Training time {training_time} seconds")
print (f"Prediction time {prediction_time} seconds")
print (f"\n {er} | {er/d} de batchs perdidos.")


mypath = args.output+f'+{datetime.now().strftime("%Y%m%d_%H%M%S")}'
if not isdir(mypath):
   makedirs(mypath)

with open(mypath+"/time.csv", "w", encoding="utf-8") as f:
    f.write("encoding_time,training_time,prediction_time,model_size\n")
    f.write(f"{encoding_time},{training_time},{prediction_time},{f'{model_size}'}")

with open(mypath+"/model.pkl", 'wb') as file:
    pkl.dump(model, file)
with open(mypath+"/results_df.pkl", 'wb') as file:
    pkl.dump(results, file)
with open(mypath+"/test_features_df.pkl", 'wb') as file:
    pkl.dump(test_features, file)
