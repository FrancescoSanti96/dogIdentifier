#!/usr/bin/env python3
"""
Helper per la configurazione del progetto Dog Breed Identifier
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigHelper:
    """Classe semplificata per caricare configurazioni e profili"""

    def __init__(self, config_path: str = "config.json"):
        """
        Inizializza l'helper di configurazione

        Args:
            config_path: Percorso al file JSON di configurazione
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

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
        Ottieni un valore usando la notazione punto (es. 'data.batch_size')

        Args:
            key: Chiave con notazione punto
            default: Valore di default se non trovato

        Returns:
            Valore o default
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_augmentation_config(self) -> Dict[str, Any]:
        """Ottieni configurazione data augmentation"""
        return self.config.get("augmentation", {})

    def apply_profile(self, profile_name: str) -> bool:
        """
        Applica un profilo come environment variables

        Args:
            profile_name: Nome del profilo da applicare

        Returns:
            True se profilo trovato e applicato, False altrimenti
        """
        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            return False

        profile = profiles[profile_name]
        print(f"🔧 Applicando profilo '{profile_name}':")

        for key, value in profile.items():
            os.environ[key] = str(value)
            print(f"   {key} = {value}")

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
