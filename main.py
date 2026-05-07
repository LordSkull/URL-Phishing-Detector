# Il dataset è stato scaricato da Kaggle, è possibile scaricarlo da questo link: https://www.kaggle.com/datasets/hasibur013/phishing-data/data
# Il dataset contiene 11055 righe e 32 colonne, di cui 31 sono feature e 1 è la label (1 se il sito è phishing, 0 altrimenti).
# L'addestramento del modello è stato effettuato utilizzando un Random Forest Classifier, che è un algoritmo di apprendimento automatico basato su alberi decisionali.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 

# Caricamento del dataset
df = pd.read_csv('PhishingData.csv')


X = df.drop('label', axis=1)   
y = df['label']   

# Suddivisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# Creazione del modello Random Forest Classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)

df.describe()

# Addestramento del modello
model.fit(X_train, y_train)

# Valutazione del modello
x_test_pred = model.predict(X_test)
accuracy = np.mean(x_test_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')

