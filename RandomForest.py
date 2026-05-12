# Il dataset è stato scaricato da Kaggle, è possibile scaricarlo da questo link: https://www.kaggle.com/datasets/hasibur013/phishing-data/data
# La reale provenienza del dataset è https://archive.ics.uci.edu/dataset/327/phishing+websites
# Il dataset contiene 11055 righe e 32 colonne, di cui 31 sono feature e 1 è la label (1 se il sito è phishing, 0 altrimenti).
# L'addestramento del modello è stato effettuato utilizzando un Random Forest Classifier, che è un algoritmo di apprendimento automatico basato su alberi decisionali.


#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importo le robe del file plot_config: la funzione di setup, i dizionari di colori
# e colormap, e la funzione di salvataggio
from plot_config import setup_plot_style, MODEL_COLORS, MODEL_CMAPS, save_figure


# funzione che applica lo stile a tutti i grafici
setup_plot_style()

# variabile dove metto il nome del modello
# utile per dizionari MODEL_COLORS e MODEL_CMAPS ma anche per il nome sul grafico
MODEL_NAME = 'Random Forest'
#endregion 


#region caricamento dataset e pulizia
# posizione dataset quando lavoro sul fisso
percorso_dataset_pc_fisso='D:/Uni/TERZO ANNO/Intelligenza Artificiale/URL-Phishing/datasets/PhishingData.csv'

# posizione dataset quando lavoro su portatile
percorso_dataset_portatile='~/Desktop/URL-Phishing-Detector/datasets/PhishingData.csv'

# Caricamento del dataset
df = pd.read_csv(percorso_dataset_portatile, skipinitialspace=True)

# i nome delle feature hanno spazi finali
# con .str.strip() li rumuovo, in questo modo tengo bello pulito
df.columns = df.columns.str.strip()

# righe colonne
print(f"\nDimensioni del dataset: {df.shape[0]} righe, {df.shape[1]} colonne")

'''
# vedere le prima 5 rifhe
print("\nPrime 5 righe:")
print(df.head())

print("\nInformazioni sulle colonne:")
df.info()
'''
#endregion


#region preparazione X e y
# seprazione feature (x) e target (y)
# rimuovo index che non mi serve
# rimuovo Result che sarebbe il target e non deve stare tra le feature
X = df.drop(columns=['index', 'Result'])
y = df['Result']   

print(f"\nNumero di feature usate per il training: {X.shape[1]}")

#endregion


#region split train test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} righe ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]} righe ({X_test.shape[0]/len(df)*100:.1f}%)")

'''
print("\nDistribuzione classi nel training set:")
print(y_train.value_counts(normalize=True).round(4) * 100)

print("\nDistribuzione classi nel test set:")
print(y_test.value_counts(normalize=True).round(4) * 100)

# stampa di varie info
print(f"\nDimensioni training set: {X_train.shape}")
print(f"Dimensioni test set: {X_test.shape}")
'''
#endregione

#region modello random forest
# Creazione del modello Random Forest Classifier
model = RandomForestClassifier(n_estimators=80, random_state=42)

# Addestramento del modello
model.fit(X_train, y_train)

#endregione


#region valutazione
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
 
print(f"\n--- RISULTATI {MODEL_NAME.upper()} ---")
print(f"Accuracy sul test set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legittimo']))
#endregione

#region grafico confusion matrix
# L'argomento labels=[-1, 1] forza l'ordine
# delle classi: prima la riga/colonna del phishing (-1), poi del legittimo (1).
# Senza questo argomento sklearn deciderebbe l'ordine da solo e potrebbe
# scambiare le posizioni di TN/TP nel grafico.
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

# ravel() "appiattisce" la matrice 2x2 in 4 valori, che assegno con un singolo
# unpacking nell'ordine standard: TN (Veri Negativi), FP (Falsi Positivi),
# FN (Falsi Negativi), TP (Veri Positivi).
tn, fp, fn, tp = cm.ravel()
 
# interpretazione grafico
print("\nConfusion Matrix:")
print(f"  Veri Negativi  (phishing identificati):     {tn}")
print(f"  Falsi Positivi (legittimi segnalati):       {fp}")
print(f"  Falsi Negativi (phishing mancati):          {fn}")
print(f"  Veri Positivi  (legittimi identificati):    {tp}")
 

# Creo una figura e un asse: figsize=(6, 5) sovrascrive il default di
# plot_config (8, 5) perché la confusion matrix è quadrata e sta meglio
# in un rapporto più compatto.
fig, ax = plt.subplots(figsize=(6, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Phishing', 'Legittimo'])

# Disegna effettivamente la matrice nell'asse 'ax' specificato.
# - cmap: prendo la colormap dal dizionario MODEL_CMAPS, così Random Forest
#   userà sempre 'Blues' senza scriverlo a mano qui
# - values_format='d': formatta i numeri come interi (senza decimali)
# - colorbar=True: mostra la legenda di colori a destra
disp.plot(ax=ax, cmap=MODEL_CMAPS[MODEL_NAME], values_format='d', colorbar=True)


# Titolo costruito da MODEL_NAME
ax.set_title(f'Confusion Matrix - {MODEL_NAME}')
ax.set_xlabel('Predetto')
ax.set_ylabel('Reale')


# La griglia tratteggiata di plot_config dà fastidio sulla confusion matrix
# perché si sovrappone alle celle colorate. La disattivo SOLO per questo
# grafico, senza toccare lo stile globale.
ax.grid(False)  

# tight_layout sistema automaticamente margini e spaziature per evitare che
# titoli o etichette vengano tagliati.
plt.tight_layout()

# Salvataggio del grafico via plot_config
save_figure(fig, 'confusion_matrix_random_forest')


plt.show()
#endregion


#region Feature Importance

# Importanza di ogni feature durante il
# training: feature_importances_ è un array con un valore per ogni colonna
# di X. Lo trasformo in pd.Series indicizzato per nome così non perdo il
# legame tra valore e nome della feature.
importances = pd.Series(model.feature_importances_, index=X.columns)
 
# Ordino in modo CRESCENTE e prendo le ultime 10 = le 10 più importanti.
# Le ordino crescente apposta: nel grafico a barre orizzontali (barh)
# matplotlib disegna dal basso verso l'alto, quindi la più grande finisce
# in cima. Esteticamente è il modo standard di presentare una top N.
top_features = importances.sort_values(ascending=True).tail(10)
 
fig, ax = plt.subplots(figsize=(8, 6))
 
# barh = barre orizzontali. Per una top 10 di feature è meglio dell'asse
# verticale perché i nomi delle feature sono lunghi e su un asse x
# ruoterebbero o si sovrapporrebbero. In orizzontale si leggono comodamente.
# Il colore arriva da MODEL_COLORS: stesso blu della confusion matrix di RF.
ax.barh(top_features.index, top_features.values, color=MODEL_COLORS[MODEL_NAME])
ax.set_xlabel('Importanza')
ax.set_title(f'Top 10 Feature più importanti - {MODEL_NAME}')
plt.tight_layout()
save_figure(fig, 'feature_importance_random_forest')
plt.show()
 
# Stampo anche in console le top 10 in ordine DECRESCENTE (più importante
# in alto) per leggerle facilmente nell'output testuale. Round a 4 decimali
# per evitare numeri lunghissimi.
print("\nTop 10 feature più importanti:")
print(top_features.sort_values(ascending=False).round(4))
#endregion