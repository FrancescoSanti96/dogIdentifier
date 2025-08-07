#!/usr/bin/env python3
"""
🐕 Dog Breed Identifier - Intermediate Dataset Preparation

Questo script prepara un dataset intermedio di 60 razze canine selezionando
quelle con più immagini per un training di test intermedio.

Configurazione:
- 60 razze con più immagini
- Train: 70%
- Validation: 15% 
- Test: 15%
- Include sempre Australian_Shepherd_Dog

Autore: Francesco Santi
Data: Agosto 2025
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse

def load_config():
    """Carica la configurazione dal file config.json"""
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("File config.json non trovato!")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_breed_statistics(breeds_path):
    """Analizza le statistiche del dataset per razza"""
    breeds_path = Path(breeds_path)
    breed_stats = {}
    
    print("📊 Analisi dataset razze canine...")
    print("=" * 60)
    
    for breed_dir in sorted(breeds_path.iterdir()):
        if breed_dir.is_dir():
            # Conta immagini per razza
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(breed_dir.glob(ext))
            
            count = len(image_files)
            breed_stats[breed_dir.name] = {
                'count': count,
                'files': [str(f) for f in image_files]
            }
            
            print(f"🐕 {breed_dir.name:<35} | {count:>4} immagini")
    
    print("=" * 60)
    print(f"📈 TOTALE: {len(breed_stats)} razze")
    print()
    
    return breed_stats

def select_top_breeds(breed_stats, num_breeds=60, force_include=None):
    """Seleziona le top N razze con più immagini"""
    if force_include is None:
        force_include = ['Australian_Shepherd_Dog']
    
    # Ordina razze per numero di immagini (decrescente)
    sorted_breeds = sorted(breed_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    selected_breeds = {}
    
    # Prima aggiungi le razze forzate
    for breed_name in force_include:
        if breed_name in breed_stats:
            selected_breeds[breed_name] = breed_stats[breed_name]
            print(f"✅ {breed_name:<35} | {breed_stats[breed_name]['count']:>4} immagini (FORZATA)")
    
    # Poi aggiungi le top breeds (escludendo quelle già forzate)
    count = len(selected_breeds)
    for breed_name, breed_data in sorted_breeds:
        if count >= num_breeds:
            break
        
        if breed_name not in selected_breeds:
            selected_breeds[breed_name] = breed_data
            count += 1
    
    print(f"\n🎯 Selezionate {len(selected_breeds)} razze per il dataset intermedio")
    
    return selected_breeds

def create_dataset_splits(breed_stats, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Crea gli splits del dataset"""
    output_path = Path(output_path)
    
    # Crea cartelle output
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = output_path / split
        split_path.mkdir(parents=True, exist_ok=True)
    
    split_stats = defaultdict(lambda: defaultdict(int))
    breed_details = []
    
    print("\n🔄 Creazione splits dataset...")
    print("=" * 60)
    
    for breed_name, breed_data in breed_stats.items():
        total_images = breed_data['count']
        image_files = breed_data['files']
        
        # Calcola numero immagini per split
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count
        
        # Mescola randomicamente le immagini
        random.shuffle(image_files)
        
        # Crea splits
        splits_data = {
            'train': image_files[:train_count],
            'val': image_files[train_count:train_count + val_count],
            'test': image_files[train_count + val_count:]
        }
        
        # Copia file negli splits
        for split_name, files in splits_data.items():
            split_breed_dir = output_path / split_name / breed_name
            split_breed_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                src_path = Path(file_path)
                dst_path = split_breed_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                split_stats[split_name][breed_name] += 1
        
        breed_details.append({
            'breed': breed_name,
            'total': total_images,
            'train': len(splits_data['train']),
            'val': len(splits_data['val']),
            'test': len(splits_data['test'])
        })
        
        print(f"✅ {breed_name:<35} | Train: {len(splits_data['train']):>3} | Val: {len(splits_data['val']):>3} | Test: {len(splits_data['test']):>3}")
    
    return split_stats, breed_details

def save_dataset_info(split_stats, breed_details, output_path):
    """Salva informazioni sui dataset splits"""
    output_path = Path(output_path)
    
    # Calcola statistiche totali
    total_stats = {}
    for split_name in ['train', 'val', 'test']:
        total_images = sum(split_stats[split_name].values())
        total_breeds = len(split_stats[split_name])
        total_stats[split_name] = {
            'images': total_images,
            'breeds': total_breeds
        }
    
    dataset_info = {
        'creation_date': '2025-08-07',
        'dataset_type': 'intermediate',
        'total_breeds': len(breed_details),
        'splits': total_stats,
        'breed_details': breed_details,
        'split_configuration': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
    }
    
    # Salva file JSON con informazioni
    info_file = output_path / 'dataset_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    return dataset_info

def print_final_summary(dataset_info):
    """Stampa il riassunto finale"""
    print("\n" + "=" * 60)
    print("📊 RIASSUNTO FINALE DATASET INTERMEDIO")
    print("=" * 60)
    
    print(f"🎯 Razze selezionate: {dataset_info['total_breeds']}")
    print()
    
    print("📈 Distribuzione splits:")
    for split_name, stats in dataset_info['splits'].items():
        print(f"  {split_name.capitalize():<12} | {stats['images']:>5} immagini | {stats['breeds']:>3} razze")
    
    total_images = sum(stats['images'] for stats in dataset_info['splits'].values())
    print(f"  {'TOTALE':<12} | {total_images:>5} immagini")
    
    print(f"\n🎯 Dataset intermedio pronto per training su {dataset_info['total_breeds']} razze!")
    
    # Mostra le prime e ultime razze selezionate
    breeds_by_images = sorted(dataset_info['breed_details'], key=lambda x: x['total'], reverse=True)
    
    print(f"\n📊 Top 10 razze per numero di immagini:")
    for i, breed in enumerate(breeds_by_images[:10], 1):
        symbol = "⭐" if breed['breed'] == 'Australian_Shepherd_Dog' else "🐕"
        print(f"  {i:>2}. {symbol} {breed['breed']:<30} | {breed['total']:>3} immagini")
    
    if len(breeds_by_images) > 10:
        print(f"\n📊 Bottom 5 razze per numero di immagini:")
        for i, breed in enumerate(breeds_by_images[-5:], len(breeds_by_images)-4):
            symbol = "⭐" if breed['breed'] == 'Australian_Shepherd_Dog' else "🐕"
            print(f"  {i:>2}. {symbol} {breed['breed']:<30} | {breed['total']:>3} immagini")

def main():
    parser = argparse.ArgumentParser(description='Prepara il dataset intermedio (60 razze)')
    parser.add_argument('--breeds-path', default='data/breeds', help='Path al dataset delle razze')
    parser.add_argument('--output-path', default='data/intermediate_splits', help='Path output per gli splits')
    parser.add_argument('--num-breeds', type=int, default=60, help='Numero di razze da selezionare')
    parser.add_argument('--seed', type=int, default=42, help='Seed per reproducibilità')
    
    args = parser.parse_args()
    
    # Imposta seed per reproducibilità
    random.seed(args.seed)
    
    print("🐕 DOG BREED IDENTIFIER - PREPARAZIONE DATASET INTERMEDIO")
    print("=" * 60)
    print(f"📁 Dataset path: {args.breeds_path}")
    print(f"📁 Output path: {args.output_path}")
    print(f"🎯 Numero razze: {args.num_breeds}")
    print(f"🎲 Random seed: {args.seed}")
    print()
    
    try:
        # Carica configurazione
        config = load_config()
        
        # Analizza statistiche dataset
        breed_stats = get_breed_statistics(args.breeds_path)
        
        # Seleziona top breeds
        selected_breeds = select_top_breeds(breed_stats, args.num_breeds)
        
        # Crea splits
        split_stats, breed_details = create_dataset_splits(selected_breeds, args.output_path)
        
        # Salva informazioni dataset
        dataset_info = save_dataset_info(split_stats, breed_details, args.output_path)
        
        # Stampa riassunto finale
        print_final_summary(dataset_info)
        
        print(f"\n✅ Dataset intermedio preparato con successo!")
        print(f"📁 Files salvati in: {args.output_path}/")
        print(f"📊 Info dataset salvate in: {args.output_path}/dataset_info.json")
        
    except Exception as e:
        print(f"❌ Errore durante la preparazione del dataset: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
