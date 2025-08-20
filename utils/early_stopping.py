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
        # Convertiamo loss in score per logica "più alto = migliore"
        # Loss: minore è meglio → Score: maggiore è meglio (-val_loss)
        score = -val_loss

        # Prima epoca: inizializzazione senza confronti
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            return False  # Mai fermare alla prima epoca

        # LOGICA CORE: controlla miglioramento vs patience
        # Se score attuale < best_score + delta → NO miglioramento significativo
        if score < self.best_score + self.delta:
            # Patience Logic: incrementa contatore epoche senza progresso
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            # Triggering: se raggiungiamo patience limit → STOP TRAINING
            if self.counter >= self.patience:
                self.early_stop = True
                return True  # Segnala training loop di fermarsi
        else:
            # IMPROVEMENT DETECTED: reset tutto e continua training
            self.best_score = score  # Aggiorna miglior score
            self.val_loss_min = val_loss  # Aggiorna miglior validation loss
            self.counter = 0  # Reset patience counter

        return False
