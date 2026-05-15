"""
Configurazione matplotlib
 
Lo scopo di questo file è garantire che tutti i grafici prodotti
dagli script abbiano lo stesso stile quindi stessi font, stessi colori e stesse dimensioni. 
"""
import os
import matplotlib.pyplot as plt

# dizionari colori

# Colori per ogni modello
MODEL_COLORS = {
    'Random Forest': '#2E86AB',   # blu
    'Decision Tree': '#06A77D',   # verde
    'KNN':           '#D62246',   # rosso
}

# per la confusion matrix matplotlib usa colormap che sarebbe una sfumatura
# ogni modello ha la propria
MODEL_CMAPS = {
    'Random Forest': 'Blues',
    'Decision Tree': 'Greens',
    'KNN':           'Reds',
}

# in questa funzione definisco lo tile di tutti i grafici
def setup_plot_style():
    """Applica lo stile a tutti i grafici."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,

        # Layout pulito
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        # Figure
        'figure.figsize': (8, 5),
        'figure.dpi': 100,
        'savefig.dpi': 300,          
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })

# funzione che salva la figura in formato png nella cartella figures
def save_figure(fig, filename, folder='figures'):
    """Salvataggio figura in png"""
    os.makedirs(folder, exist_ok=True)
    path = f'{folder}/{filename}.png'
    fig.savefig(path)
    print(f"Grafico salvato: {path}")