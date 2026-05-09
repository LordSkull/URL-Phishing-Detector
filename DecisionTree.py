# Il dataset è stato scaricato da Kaggle, è possibile scaricarlo da questo link: https://www.kaggle.com/datasets/hasibur013/phishing-data/data
# La reale provenienza del dataset è https://archive.ics.uci.edu/dataset/327/phishing+websites
# Il dataset contiene 11055 righe e 32 colonne, di cui 31 sono feature e 1 è la label (1 se il sito è phishing, 0 altrimenti).
# L'addestramento del modello è stato effettuato utilizzando un Decision Tree Classifier

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# 0. Setup logging: tutto ciò che viene stampato con print() finisce sia
#    a schermo sia in un file di testo, per poterlo rileggere con calma.
# -----------------------------------------------------------------------------
class Tee:
    """Duplica l'output su più stream"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# Crea la cartella logs/ se non esiste
os.makedirs('logs', exist_ok=True)

# Apre il file di log con un timestamp per non sovrascrivere esecuzioni precedenti
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f'output/decision_tree_{timestamp}.txt'
log_file = open(log_path, 'w', encoding='utf-8')

# Sostituisce sys.stdout con il nostro Tee: da qui in poi ogni print() va
# sia a schermo sia nel file
sys.stdout = Tee(sys.stdout, log_file)

print(f"Esecuzione avviata: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log salvato in: {log_path}\n")

# -----------------------------------------------------------------------------
# 1. Caricamento e pulizia
# -----------------------------------------------------------------------------

# posizione dataset quando lavoro sul fisso
percorso_dataset_pc_fisso='D:/Uni/TERZO ANNO/Intelligenza Artificiale/URL-Phishing/datasets/PhishingData.csv'

# posizione dataset quando lavoro su portatile
percorso_dataset_portatile='~/Desktop/URL-Phishing-Detector/datasets/PhishingData.csv'

# Caricamento del dataset
df = pd.read_csv(percorso_dataset_pc_fisso, skipinitialspace=True)
df.columns = df.columns.str.strip()


print(f"\nDataset: {df.shape[0]} righe, {df.shape[1]} colonne")

# -----------------------------------------------------------------------------
# 2. Preparazione X e y
# -----------------------------------------------------------------------------

X = df.drop(columns=['index', 'Result'])
y = df['Result']


# -----------------------------------------------------------------------------
# 3. Split train/test
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 4. Modello Decision Tree
# -----------------------------------------------------------------------------
model = DecisionTreeClassifier(criterion='gini', random_state=42)

model.fit(X_train, y_train)


# -----------------------------------------------------------------------------
# 5. Valutazione
# -----------------------------------------------------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- RISULTATI ---")
print(f"Accuracy sul test set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legittimo']))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Profondità effettiva dell'albero: dato interessante per capire la complessità
print(f"\nProfondità albero: {model.get_depth()}")
print(f"Numero foglie: {model.get_n_leaves()}")


# -----------------------------------------------------------------------------
# 6. Grafico Confusion Matrix
# -----------------------------------------------------------------------------
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

print("Confusion Matrix:")
print(cm)

# Grafico Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Phishing', 'Legittimo']
)
disp.plot(ax=ax, cmap='Greens', values_format='d', colorbar=True)
ax.set_title('Confusion Matrix - Decision Tree', fontsize=13)
plt.tight_layout()
plt.show()