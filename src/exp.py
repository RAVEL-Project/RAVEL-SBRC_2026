import torch
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

#git hub e python setup.py install
from torchwnn.datasets.iris import Iris
#from sklearn.datasets import load_iris
from torchwnn.classifiers import Wisard
from torchwnn.encoding import Thermometer

# ==================================================
# Device
# ==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ==================================================
# Load dataset
# ==================================================

#iris = load_iris()#
iris=Iris()
X = torch.tensor(iris.features.values).to(device)
y = torch.tensor(list(iris.labels)).squeeze().to(device)

# ==================================================
# FIXED train / test split
# ==================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# ==================================================
# Values to test
# ==================================================
bits_values = [5, 10, 15, 20, 25, 30]

results = []

for bits_encoding in bits_values:
    print(f"\nRunning experiment with bits_encoding = {bits_encoding}")

    # ==================================================
    # Encoding
    # ==================================================
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_encoding = time.perf_counter()

    encoding = Thermometer(bits_encoding).fit(X_train_raw)
    X_train = encoding.binarize(X_train_raw).flatten(start_dim=1)
    X_test = encoding.binarize(X_test_raw).flatten(start_dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize()
    encoding_time = time.perf_counter() - start_encoding

    # ==================================================
    # Model
    # ==================================================
    entry_size = X_train.shape[1]
    model = Wisard(entry_size, iris.num_classes, tuple_size=8)

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