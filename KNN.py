# Il dataset è stato scaricato da Kaggle, è possibile scaricarlo da questo link: https://www.kaggle.com/datasets/hasibur013/phishing-data/data
# La reale provenienza del dataset è https://archive.ics.uci.edu/dataset/327/phishing+websites
# Il dataset contiene 11055 righe e 32 colonne, di cui 31 sono feature e 1 è la label (1 se il sito è phishing, 0 altrimenti).
# L'addestramento del modello è stato effettuato utilizzando KNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('D:/Uni/TERZO ANNO/Intelligenza Artificiale/URL-Phishing/datasets/PhishingData.csv', skipinitialspace=True)
df.columns = df.columns.str.strip()


print(f"\nDataset: {df.shape[0]} righe, {df.shape[1]} colonne")


X = df.drop(columns=['index', 'Result'])
y = df['Result']


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
