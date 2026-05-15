# Il dataset è stato scaricato da Kaggle, è possibile scaricarlo da questo link: https://www.kaggle.com/datasets/hasibur013/phishing-data/data
# La reale provenienza del dataset è https://archive.ics.uci.edu/dataset/327/phishing+websites
# Il dataset contiene 11055 righe e 32 colonne, di cui 31 sono feature e 1 è la label (1 se il sito è phishing, -1 altrimenti).
# L'addestramento del modello è stato effettuato utilizzando KNN
# Per dividere il codice in sezione ho usate il region che è nativo di VisualStudioCode


# ──────────────────────────────────────────────────────────────────────────────
#region IMPORT
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importo le robe del file plot_config: la funzione di setup, i dizionari di colori
# e colormap, e la funzione di salvataggio
from plot_config import setup_plot_style, MODEL_COLORS, MODEL_CMAPS, save_figure

# funzione che applica lo stile a tutti i grafici
setup_plot_style()

# variabile dove metto il nome del modello
# utile per dizionari MODEL_COLORS e MODEL_CMAPS ma anche per il nome sul grafico
MODEL_NAME = 'KNN'
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Caricamento e pulizia
# ──────────────────────────────────────────────────────────────────────────────
# posizione dataset quando lavoro sul fisso
percorso_dataset_pc_fisso='D:/Uni/TERZO ANNO/Intelligenza Artificiale/URL-Phishing/datasets/PhishingData.csv'

# posizione dataset quando lavoro su portatile
percorso_dataset_portatile='~/Desktop/URL-Phishing-Detector/datasets/PhishingData.csv'

# Caricamento del dataset
df = pd.read_csv(percorso_dataset_pc_fisso, skipinitialspace=True)
df.columns = df.columns.str.strip()


print(f"\nDimensioni del Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Preparazione X e Y
# ──────────────────────────────────────────────────────────────────────────────

# separazione feature (x) e target (y)
# rimuovo index che non mi serve
# rimuovo Result che sarebbe il target e non deve stare tra le feature
X = df.drop(columns=['index', 'Result'])
y = df['Result']
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Split Train e test set
# ──────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Training set: {X_train.shape[0]} righe ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]} righe ({X_test.shape[0]/len(df)*100:.1f}%)")

#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Modello KNN con K non ottimale
# ──────────────────────────────────────────────────────────────────────────────

# n_neighbors=5: guarda i 5 vicini più vicini per votare la classe.
# È il default di sklearn ed è un valore equilibrato. 
# Dopo provo altri valori di K per vedere quale funziona meglio.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Valutazione
# ──────────────────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- RISULTATI {MODEL_NAME.upper()} (k=5) ---")
print(f"Accuracy sul test set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legittimo']))
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

# L'argomento labels=[-1, 1] forza l'ordine
# delle classi: prima la riga/colonna del phishing (-1), poi del legittimo (1).
# Senza questo argomento sklearn deciderebbe l'ordine da solo 
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

# ravel() "appiattisce" la matrice 2x2 in 4 valori, che assegno con un singolo
# unpacking nell'ordine standard: TN (Veri Negativi), FP (Falsi Positivi),
# FN (Falsi Negativi), TP (Veri Positivi).
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"  Veri Negativi  (phishing identificati):     {tn}")
print(f"  Falsi Positivi (legittimi segnalati):       {fp}")
print(f"  Falsi Negativi (phishing mancati):          {fn}")
print(f"  Veri Positivi  (legittimi identificati):    {tp}")

# Creo una figura e un asse: figsize=(6, 5) sovrascrivo il default di
# plot_config (8, 5) perché con questo rapporto sta meglio
fig, ax = plt.subplots(figsize=(6, 5))


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Phishing', 'Legittimo'])


# Disegna effettivamente la matrice nell'asse 'ax' specificato.
# - cmap: prendo la colormap dal dizionario MODEL_CMAPS, così KNN
#   userà sempre 'Reds' senza scriverlo a mano qui
# - values_format='d': formatta i numeri come interi
# - colorbar=True: mostra la legenda di colori a destra
disp.plot(ax=ax, cmap=MODEL_CMAPS[MODEL_NAME], values_format='d', colorbar=True)

# Titolo costruito da MODEL_NAME
ax.set_title(f'Confusion Matrix - {MODEL_NAME}')
ax.set_xlabel('Predetto')
ax.set_ylabel('Reale')

# Disattivo la La griglia tratteggiata di plot_config che dà fastidio sulla confusion matrix
ax.grid(False)


# tight_layout sistema automaticamente margini e spaziature per evitare che
# titoli o etichette vengano tagliati.
plt.tight_layout()

# Salvataggio del grafico via plot_config
save_figure(fig, 'confusion_matrix_knn')

plt.show()
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Ricerca del K ottimale
# ──────────────────────────────────────────────────────────────────────────────

# Addestro KNN con diversi valori di K e confronto le accuracy
# sul test set. Uso solo numeri DISPARI per evitare pareggi nel voto (con K
# pari, si può avere ambiguità).

print("\n--- RICERCA DEL K MIGLIORE ---")

k_values = [1, 3, 5, 7, 9, 11, 15]
k_scores = []  # qui salvo le accuracy corrispondenti

for k in k_values:
    # Creo un KNN con il K corrente
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    # Addestro sul training set
    knn_temp.fit(X_train, y_train)
    # Predico sul test set
    pred_temp = knn_temp.predict(X_test)
    # Calcolo accuracy
    acc_temp = accuracy_score(y_test, pred_temp)
    k_scores.append(acc_temp)
    print(f"  k={k:3d}  →  Accuracy = {acc_temp:.4f}")

# Trovo il K che ha dato l'accuracy massima.
# max(k_scores) prende il valore più alto, k_scores.index(...) trova la sua
# posizione nella lista, e con quella posizione prendo il K corrispondente.
best_score = max(k_scores)
best_k = k_values[k_scores.index(best_score)]
print(f"\n  K migliore: {best_k} con accuracy {best_score:.4f}")
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Grafico K vs Accuracy
# ──────────────────────────────────────────────────────────────────────────────

# Curva che mostra come varia l'accuracy al variare di K.
# A sinistra del picco c'è overfitting (K troppo piccolo, troppo rumoroso),
# a destra c'è underfitting (K troppo grande, media troppi vicini).

fig, ax = plt.subplots(figsize=(9, 5))

# I marker rendono visibili i singoli K testati.
# Colore preso da MODEL_COLORS per coerenza con gli altri grafici di KNN.
ax.plot(k_values, k_scores, marker='o', linewidth=2, color=MODEL_COLORS[MODEL_NAME], markersize=8)

# Linea verticale tratteggiata sul K ottimale per evidenziarlo visivamente.
ax.axvline(best_k, color='black', linestyle='--', alpha=0.5,label=f'K ottimale = {best_k}')

ax.set_title(f'Accuracy {MODEL_NAME} al variare di K')
ax.set_xlabel('Numero di vicini (K)')
ax.set_ylabel('Accuracy (cross-validation)')
ax.legend()

plt.tight_layout()
save_figure(fig, 'k_vs_accuracy_knn')
plt.show()
#endregion

# ──────────────────────────────────────────────────────────────────────────────
#region Modello finale con K ottimale
# ──────────────────────────────────────────────────────────────────────────────

# Riaddestro KNN con il K migliore trovato e valuto sul test set.
# Questo è il modello finale di KNN che porto al confronto con gli altri.

print(f"\n--- MODELLO FINALE {MODEL_NAME.upper()} (k={best_k}) ---")

best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f"Accuracy sul test set: {best_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Phishing', 'Legittimo']))

# Confusion matrix del modello con K migliore
cm_best = confusion_matrix(y_test, y_pred_best, labels=[-1, 1])
tn, fp, fn, tp = cm_best.ravel()

print("\nConfusion Matrix (K migliore):")
print(f"  Veri Negativi  (phishing identificati):     {tn}")
print(f"  Falsi Positivi (legittimi segnalati):       {fp}")
print(f"  Falsi Negativi (phishing mancati):          {fn}")
print(f"  Veri Positivi  (legittimi identificati):    {tp}")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=['Phishing', 'Legittimo'])
disp.plot(ax=ax, cmap=MODEL_CMAPS[MODEL_NAME], values_format='d', colorbar=True)
ax.set_title(f'Confusion Matrix - {MODEL_NAME} (k={best_k})')
ax.set_xlabel('Predetto')
ax.set_ylabel('Reale')
ax.grid(False)
plt.tight_layout()
save_figure(fig, 'confusion_matrix_knn_migliore')
plt.show()
#endregion

