 #!/usr/bin/env python3
"""
Script per rinominare le immagini Australian Shepherd con nomi puliti
"""

import os
import shutil
from pathlib import Path
import re

def rename_australian_images():
    """Rinomina tutte le immagini Australian Shepherd"""
    print("🔄 Rinominando immagini Australian Shepherd...")
    
    folder_path = Path("data/breeds/Australian_Shepherd_Dog")
    
    if not folder_path.exists():
        print("❌ Cartella Australian_Shepherd_Dog non trovata!")
        return
    
    # Lista tutti i file
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(folder_path.glob(ext))
    
    print(f"📁 Trovate {len(image_files)} immagini da rinominare...")
    
    # Contatore per nomi
    counter = 1
    
    for img_file in image_files:
        # Estrai estensione
        ext = img_file.suffix.lower()
        
        # Salta file .webp (non supportati)
        if ext == '.webp':
            print(f"⚠️  Saltando {img_file.name} (formato .webp non supportato)")
            continue
        
        # Crea nuovo nome
        new_name = f"australian_shepherd_{counter:03d}{ext}"
        new_path = folder_path / new_name
        
        # Evita conflitti
        while new_path.exists():
            counter += 1
            new_name = f"australian_shepherd_{counter:03d}{ext}"
            new_path = folder_path / new_name
        
        # Rinomina
        try:
            img_file.rename(new_path)
            print(f"✅ {img_file.name} → {new_name}")
            counter += 1
        except Exception as e:
            print(f"❌ Errore rinominando {img_file.name}: {e}")
    
    print(f"✅ Rinominazione completata! {counter-1} immagini processate.")

if __name__ == "__main__":
    rename_australian_images()