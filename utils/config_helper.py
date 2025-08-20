#!/usr/bin/env python3
"""
Helper per la configurazione del progetto Dog Breed Identifier
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigHelper:
    """
    Helper semplificato per gestione configurazione centralizzata

    Funzionalità principali:
    - Caricamento config.json con fallback sicuri
    - Sistema profili per esperimenti diversi
    - Dot notation access (es. "training.learning_rate")
    - Auto-creazione directory output
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Inizializza l'helper di configurazione

        Args:
            config_path: Percorso al file JSON di configurazione
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()  # Carica immediatamente la configurazione

    def _load_config(self) -> Dict[str, Any]:
        """Carica la configurazione dal file JSON"""
        if not self.config_path.exists():
            # Se non esiste, ritorna config vuoto invece di errore
            return {}

        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Errore nel parsing di {self.config_path}, usando config vuoto")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Accesso Dot Notation - naviga nested dictionaries con sintassi semplice

        Esempi:
        - "training.learning_rate" → config["training"]["learning_rate"]
        - "data.batch_size" → config["data"]["batch_size"]
        - "augmentation.horizontal_flip" → config["augmentation"]["horizontal_flip"]

        Args:
            key: Chiave con notazione punto per navigazione nested
            default: Valore di default se non trovato (None se omesso)

        Returns:
            Valore trovato o default
        """
        keys = key.split(".")  # Split della stringa: "a.b.c" → ["a", "b", "c"]
        value = self.config

        try:
            # Naviga la nested structure step by step
            for k in keys:
                value = value[k]  # value = value["a"], poi value = value["b"], ecc.
            return value
        except (KeyError, TypeError):
            return default  # Fallback sicuro se key non esiste o struttura wrong

    def get_augmentation_config(self) -> Dict[str, Any]:
        """Configurazione Data Augmentation - shortcut per sezione importante"""
        return self.config.get("augmentation", {})

    def apply_profile(self, profile_name: str) -> bool:
        """
        Sistema Profili - configura esperimenti rapidamente via environment variables

        I profili permettono di switschare facilmente tra configurazioni:
        - "quick_test": batch piccoli, poche epoche
        - "full_training": parametri completi per production
        - "debug": verbose logging, deterministic mode

        Meccanismo: profilo → environment variables → parsing negli script

        Args:
            profile_name: Nome del profilo da applicare (deve esistere in config.json)

        Returns:
            True se profilo trovato e applicato, False se non esiste
        """
        profiles = self.config.get("profiles", {})  # Sezione "profiles" di config.json
        if profile_name not in profiles:
            return False  # Profilo non trovato

        # Applica tutte le variabili del profilo come environment variables
        profile = profiles[profile_name]
        print(f"🔧 Applicando profilo '{profile_name}':")

        for key, value in profile.items():
            os.environ[key] = str(value)  # Environment variable (sempre stringa)
            print(f"   {key} = {value}")  # Log per debugging

        return True

    def get_profile_names(self) -> list:
        """Ottieni lista nomi profili disponibili"""
        return list(self.config.get("profiles", {}).keys())

    def print_config(self):
        """Stampa la configurazione corrente"""
        print("📋 Configurazione:")
        if self.config:
            print(json.dumps(self.config, indent=2))
        else:
            print("   Nessuna configurazione caricata")


def load_config(config_path: str = "config.json") -> ConfigHelper:
    """
    Funzione di convenienza per caricare la configurazione

    Args:
        config_path: Percorso al file di configurazione

    Returns:
        Istanza di ConfigHelper
    """
    return ConfigHelper(config_path)
