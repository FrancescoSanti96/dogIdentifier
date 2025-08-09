#!/usr/bin/env python3
"""
🐕 Dog Breed Identifier - Full Dataset Preparation

Questo script prepara il dataset completo di 121 razze canine creando splits 
train/validation/test per il training finale del modello.

Configurazione:
- Train: 70%
- Validation: 15%
- Test: 15%
- Minimo 5 immagini per split per razza

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
    total_images = 0
    
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
            total_images += count
            
            print(f"🐕 {breed_dir.name:<35} | {count:>4} immagini")
    
    print("=" * 60)
    print(f"📈 TOTALE: {len(breed_stats)} razze, {total_images} immagini")
    print()
    
    return breed_stats, total_images

def create_dataset_splits(breed_stats, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_images_per_split=5):
    """Crea gli splits del dataset"""
    output_path = Path(output_path)
    
    # Crea cartelle output
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = output_path / split
        split_path.mkdir(parents=True, exist_ok=True)
    
    split_stats = defaultdict(lambda: defaultdict(int))
    excluded_breeds = []
    included_breeds = []
    
    print("🔄 Creazione splits dataset...")
    print("=" * 60)
    
    for breed_name, breed_data in breed_stats.items():
        total_images = breed_data['count']
        image_files = breed_data['files']
        
        # Calcola numero immagini per split
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count
        
        # Verifica che ogni split abbia almeno il minimo richiesto
        if train_count < min_images_per_split or val_count < min_images_per_split or test_count < min_images_per_split:
            excluded_breeds.append({
                'breed': breed_name,
                'total': total_images,
                'train': train_count,
                'val': val_count,
                'test': test_count
            })
            print(f"❌ {breed_name:<35} | {total_images:>3} immagini (ESCLUSA - troppo poche)")
            continue
        
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
        
        included_breeds.append({
            'breed': breed_name,
            'total': total_images,
            'train': len(splits_data['train']),
            'val': len(splits_data['val']),
            'test': len(splits_data['test'])
        })
        
        print(f"✅ {breed_name:<35} | Train: {len(splits_data['train']):>3} | Val: {len(splits_data['val']):>3} | Test: {len(splits_data['test']):>3}")
    
    return split_stats, included_breeds, excluded_breeds

def save_dataset_info(split_stats, included_breeds, excluded_breeds, output_path):
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
        'total_breeds_processed': len(included_breeds) + len(excluded_breeds),
        'included_breeds': len(included_breeds),
        'excluded_breeds': len(excluded_breeds),
        'splits': total_stats,
        'included_breeds_details': included_breeds,
        'excluded_breeds_details': excluded_breeds,
        'split_configuration': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'min_images_per_split': 5
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
    print("📊 RIASSUNTO FINALE DATASET")
    print("=" * 60)
    
    print(f"🎯 Razze totali processate: {dataset_info['total_breeds_processed']}")
    print(f"✅ Razze incluse: {dataset_info['included_breeds']}")
    print(f"❌ Razze escluse: {dataset_info['excluded_breeds']} (troppe poche immagini)")
    print()
    
    print("📈 Distribuzione splits:")
    for split_name, stats in dataset_info['splits'].items():
        print(f"  {split_name.capitalize():<12} | {stats['images']:>5} immagini | {stats['breeds']:>3} razze")
    
    total_images = sum(stats['images'] for stats in dataset_info['splits'].values())
    print(f"  {'TOTALE':<12} | {total_images:>5} immagini")
    
    print(f"\n🎯 Dataset pronto per training su {dataset_info['included_breeds']} razze!")
    
    if dataset_info['excluded_breeds'] > 0:
        print(f"\n⚠️  NOTA: {dataset_info['excluded_breeds']} razze escluse per dataset insufficiente")
        print("   (meno di 5 immagini per train/val/test)")

def main():
    parser = argparse.ArgumentParser(description='Prepara il dataset completo per il training')
    parser.add_argument('--breeds-path', default='data/breeds', help='Path al dataset delle razze')
    parser.add_argument('--output-path', default='data/full_splits', help='Path output per gli splits')
    parser.add_argument('--seed', type=int, default=42, help='Seed per reproducibilità')
    parser.add_argument('--min-images', type=int, default=5, help='Minimo immagini per split')
    
    args = parser.parse_args()
    
    # Imposta seed per reproducibilità
    random.seed(args.seed)
    
    print("🐕 DOG BREED IDENTIFIER - PREPARAZIONE DATASET COMPLETO")
    print("=" * 60)
    print(f"📁 Dataset path: {args.breeds_path}")
    print(f"📁 Output path: {args.output_path}")
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Minimo immagini per split: {args.min_images}")
    print()
    
    try:
        # Carica configurazione
        config = load_config()
        
        # Analizza statistiche dataset
        breed_stats, total_images = get_breed_statistics(args.breeds_path)
        
        # Crea splits
        split_stats, included_breeds, excluded_breeds = create_dataset_splits(
            breed_stats, 
            args.output_path,
            min_images_per_split=args.min_images
        )
        
        # Salva informazioni dataset
        dataset_info = save_dataset_info(split_stats, included_breeds, excluded_breeds, args.output_path)
        
        # Stampa riassunto finale
        print_final_summary(dataset_info)
        
        print(f"\n✅ Dataset preparato con successo!")
        print(f"📁 Files salvati in: {args.output_path}/")
        print(f"📊 Info dataset salvate in: {args.output_path}/dataset_info.json")
        
    except Exception as e:
        print(f"❌ Errore durante la preparazione del dataset: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
