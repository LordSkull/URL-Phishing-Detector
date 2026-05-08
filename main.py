# Il dataset è stato scaricato da Kaggle, è possibile scaricarlo da questo link: https://www.kaggle.com/datasets/hasibur013/phishing-data/data
# La reale provenienza del dataset è https://archive.ics.uci.edu/dataset/327/phishing+websites
# Il dataset contiene 11055 righe e 32 colonne, di cui 31 sono feature e 1 è la label (1 se il sito è phishing, 0 altrimenti).
# L'addestramento del modello è stato effettuato utilizzando un Random Forest Classifier, che è un algoritmo di apprendimento automatico basato su alberi decisionali.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Caricamento del dataset
df = pd.read_csv('D:/Uni/TERZO ANNO/Intelligenza Artificiale/URL-Phishing/datasets/PhishingData.csv', skipinitialspace=True)

# posizione dataset quando lavoro su portatile
#df = pd.read_csv('~/Desktop/URL-Phishing-Detector/datasets/PhishingData.csv', skipinitialspace=True)

# i nome delle feature hanno spazi finali
# con .str.strip() li rumuovo, in questo modo tengo bello pulito
df.columns = df.columns.str.strip()

# righe colonne
print(f"\nDimensioni del dataset: {df.shape[0]} righe, {df.shape[1]} colonne")

# vedere le prima 5 rifhe
print("\nPrime 5 righe:")
print(df.head())

print("\nInformazioni sulle colonne:")
df.info()

# seprazione feature (x) e target (y)
# rimuovo index che non mi serve
# rimuovo Result che sarebbe il target e non deve stare tra le feature
X = df.drop(columns=['index', 'Result'])
y = df['Result']   

print(f"\nNumero di feature usate per il training: {X.shape[1]}")

# Suddivisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

print(f"\nDimensioni training set: {X_train.shape[0]} righe ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Dimensioni test set:     {X_test.shape[0]} righe ({X_test.shape[0]/len(df)*100:.1f}%)")

print("\nDistribuzione classi nel training set:")
print(y_train.value_counts(normalize=True).round(4) * 100)

print("\nDistribuzione classi nel test set:")
print(y_test.value_counts(normalize=True).round(4) * 100)



# stampa di varie info
print(f"\nDimensioni training set: {X_train.shape}")
print(f"Dimensioni test set: {X_test.shape}")

# Creazione del modello Random Forest Classifier
model = RandomForestClassifier(n_estimators=80, random_state=42)

# Addestramento del modello
model.fit(X_train, y_train)

# valutazione sul test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nRISULTATI SUL TEST SET")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Phishing (-1)', 'Legittimo (1)']))

# Valutazione del modello
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Risultati sul test set ---")
print(f"Accuracy: {accuracy:.4f}")


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

# Grafico Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Phishing', 'Legittimo']
)
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
ax.set_title('Confusion Matrix - Random Forest', fontsize=13)
ax.set_xlabel('Predetto')
ax.set_ylabel('Reale')
plt.tight_layout()
plt.show()


# interpretazione dati grafico
tn, fp, fn, tp = cm.ravel()
print(f"\nInterpretazione confusion matrix:")
print(f"  Veri Negativi (phishing correttamente identificato): {tn}")
print(f"  Falsi Positivi (legittimo segnalato come phishing):  {fp}")
print(f"  Falsi Negativi (phishing mancato, molto pericoloso): {fn}")
print(f"  Veri Positivi (legittimo correttamente identificato): {tp}")