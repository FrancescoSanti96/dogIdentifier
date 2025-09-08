#!/usr/bin/env python3
"""
Dataset Simulator - Modifica temporaneamente il dataset per replicare processo storico
Backup + Riduce dataset per simulare condizioni originali del PROCESSO.md
"""

import os
import shutil
import random
from pathlib import Path

def backup_original_dataset(source_dir="data/my_dog_vs_others", 
                          backup_dir="data/my_dog_vs_others_BACKUP"):
    """Crea backup del dataset completo"""
    if os.path.exists(backup_dir):
        print(f"⚠️ Backup già esistente: {backup_dir}")
        return False
    
    print(f"📦 Creando backup dataset completo...")
    shutil.copytree(source_dir, backup_dir)
    print(f"✅ Backup salvato in: {backup_dir}")
    return True

def create_reduced_dataset(source_dir="data/my_dog_vs_others", 
                         target_total=272):
    """
    Riduce il dataset esistente per simulare condizioni originali
    
    Dataset originale documentato (PROCESSO.md):
    - Total: 272 immagini
    - Maggie: ~136 immagini  
    - Others: ~136 immagini
    
    Strategia: 
    1. Backup dataset completo
    2. Rimuovi immagini casuali per raggiungere target
    3. Mantieni bilanciamento classi
    """
    
    print(f"� Riducendo dataset a {target_total} immagini (simulazione originale)...")
    
    # Target per classe (bilanciato)
    target_per_class = target_total // 2  # 136 per classe
    
    # Processa Maggie
    maggie_dir = Path(f"{source_dir}/maggie")
    maggie_files = list(maggie_dir.glob("*"))
    current_maggie = len(maggie_files)
    
    if current_maggie > target_per_class:
        # Rimuovi immagini casuali (mantieni seed per riproducibilità)
        random.seed(42)
        to_remove = random.sample(maggie_files, current_maggie - target_per_class)
        for file in to_remove:
            file.unlink()
        print(f"🗑️ Maggie: {current_maggie} → {target_per_class} immagini")
    else:
        print(f"✅ Maggie: {current_maggie} immagini (già sotto target)")
    
    # Processa Others
    other_dir = Path(f"{source_dir}/other")
    other_files = list(other_dir.glob("*"))
    current_other = len(other_files)
    
    if current_other > target_per_class:
        random.seed(43)  # Seed diverso per variety
        to_remove = random.sample(other_files, current_other - target_per_class)
        for file in to_remove:
            file.unlink()
        print(f"🗑️ Others: {current_other} → {target_per_class} immagini")
    else:
        print(f"✅ Others: {current_other} immagini (già sotto target)")
    
    # Verifica finale
    final_maggie = len(list(maggie_dir.glob("*")))
    final_other = len(list(other_dir.glob("*")))
    final_total = final_maggie + final_other
    
    print(f"📊 Dataset ridotto creato:")
    print(f"   ├── Maggie: {final_maggie} immagini")
    print(f"   ├── Others: {final_other} immagini") 
    print(f"   └── Total: {final_total} immagini")
    
    return final_total

def restore_original_dataset(source_dir="data/my_dog_vs_others",
                           backup_dir="data/my_dog_vs_others_BACKUP"):
    """Ripristina dataset completo dal backup"""
    if not os.path.exists(backup_dir):
        print(f"❌ Error: Backup non trovato: {backup_dir}")
        return False
    
    print(f"🔄 Ripristinando dataset completo dal backup...")
    
    # Rimuovi directory corrente
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    
    # Ripristina da backup
    shutil.copytree(backup_dir, source_dir)
    
    # Verifica ripristino
    maggie_count = len(list(Path(f"{source_dir}/maggie").glob("*")))
    other_count = len(list(Path(f"{source_dir}/other").glob("*")))
    total = maggie_count + other_count
    
    print(f"✅ Dataset ripristinato:")
    print(f"   ├── Maggie: {maggie_count} immagini")
    print(f"   ├── Others: {other_count} immagini")
    print(f"   └── Total: {total} immagini")
    
    # Rimuovi backup
    shutil.rmtree(backup_dir)
    print(f"🗑️ Backup rimosso: {backup_dir}")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simula dataset originale per replicare processo")
    parser.add_argument("--reduce", action="store_true", help="Riduci dataset a dimensioni originali")
    parser.add_argument("--restore", action="store_true", help="Ripristina dataset completo")
    parser.add_argument("--target", type=int, default=272, help="Target immagini totali (default: 272)")
    
    args = parser.parse_args()
    
    if args.reduce:
        # Backup e riduzione
        if backup_original_dataset():
            create_reduced_dataset(target_total=args.target)
        else:
            print("⚠️ Backup già esistente. Usa --restore per ripristinare prima.")
    elif args.restore:
        restore_original_dataset()
    else:
        print("Usa --reduce per ridurre dataset o --restore per ripristinare")
