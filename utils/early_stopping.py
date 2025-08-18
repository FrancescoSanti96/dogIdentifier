import numpy as np


class EarlyStopping:
    """
    Early Stopping per prevenire overfitting durante il training

    Ferma il training quando la validation loss non migliora per 'patience' epoche consecutive.

    Args:
        patience (int): Numero di epoche da aspettare senza miglioramento
        verbose (bool): Stampa messaggi di debug
        delta (float): Miglioramento minimo per considerare un progresso
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience  # Epoche di pazienza
        self.verbose = verbose  # Debug output
        self.counter = 0  # Contatore epoche senza miglioramento
        self.best_score = None  # Miglior score finora (-val_loss)
        self.early_stop = False  # Flag per fermare training
        self.val_loss_min = np.inf  # Miglior validation loss (fix NumPy 2.0+)
        self.delta = delta  # Soglia minima miglioramento

    def __call__(self, val_loss):
        """
        Controlla se fermare il training basandosi sulla validation loss

        Args:
            val_loss (float): Validation loss dell'epoca corrente

        Returns:
            bool: True se il training deve essere fermato, False altrimenti
        """
        # Convertiamo loss in score (più alto = meglio)
        score = -val_loss

        # Prima epoca: inizializza best score
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return False

        # Controlla se c'è miglioramento significativo
        if score < self.best_score + self.delta:
            # Nessun miglioramento: incrementa counter
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            # Se raggiungiamo patience: ferma training
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            # Miglioramento trovato: reset counter e aggiorna best
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

        return False
