#!/usr/bin/env python3
"""
Analisi dettagliata del dataset intermedio (60 razze)
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

def analyze_dataset():
    """Analizza la distribuzione del dataset intermedio"""
    print("📊 Analisi Dataset Intermedio (60 Razze)")
    print("=" * 50)
    
    splits_dir = "data/intermediate_splits"
    splits = ['train', 'val', 'test']
    
    all_data = []
    
    for split in splits:
        split_dir = os.path.join(splits_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ {split_dir} non trovato!")
            continue
            
        print(f"\\n📁 Analizzando split: {split.upper()}")
        
        breeds = os.listdir(split_dir)
        breeds = [b for b in breeds if os.path.isdir(os.path.join(split_dir, b))]
        breeds.sort()
        
        split_data = []
        for breed in breeds:
            breed_dir = os.path.join(split_dir, breed)
            images = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            split_data.append({'breed': breed, 'split': split, 'count': count})
            all_data.append({'breed': breed, 'split': split, 'count': count})
        
        # Statistiche per questo split
        counts = [d['count'] for d in split_data]
        print(f"   📊 Totale razze: {len(breeds)}")
        print(f"   📊 Totale immagini: {sum(counts)}")
        print(f"   📊 Media per razza: {sum(counts)/len(counts):.1f}")
        print(f"   📊 Min-Max: {min(counts)}-{max(counts)}")
        print(f"   📊 Std dev: {pd.Series(counts).std():.1f}")
        
        # Top/Bottom 5 razze per questo split
        split_df = pd.DataFrame(split_data).sort_values('count', ascending=False)
        print(f"\\n   🏆 TOP 5 razze ({split}):")
        for _, row in split_df.head().iterrows():
            print(f"      {row['breed']}: {row['count']} images")
        
        print(f"\\n   🔽 BOTTOM 5 razze ({split}):")
        for _, row in split_df.tail().iterrows():
            print(f"      {row['breed']}: {row['count']} images")
    
    # Analisi complessiva
    print(f"\\n" + "=" * 60)
    print(f"📈 ANALISI COMPLESSIVA")
    print(f"=" * 60)
    
    df = pd.DataFrame(all_data)
    
    # Pivot table per vedere distribuzione
    pivot = df.pivot(index='breed', columns='split', values='count').fillna(0)
    pivot['total'] = pivot.sum(axis=1)
    pivot_sorted = pivot.sort_values('total', ascending=False)
    
    print(f"\\n🎯 Distribuzione totale per razza:")
    print(pivot_sorted.head(10).to_string())
    
    # Controllo bilanciamento
    train_counts = pivot['train'].values
    val_counts = pivot['val'].values  
    test_counts = pivot['test'].values
    total_counts = pivot['total'].values
    
    print(f"\\n📊 Statistiche globali:")
    print(f"   📊 Razze totali: {len(pivot)}")
    print(f"   📊 Immagini totali: {total_counts.sum()}")
    print(f"   📊 Media per razza: {total_counts.mean():.1f}")
    print(f"   📊 Std dev totale: {total_counts.std():.1f}")
    print(f"   📊 Coefficient of Variation: {total_counts.std()/total_counts.mean():.3f}")
    
    print(f"\\n🎯 Controllo Australian_Shepherd_Dog:")
    aus_data = pivot_sorted[pivot_sorted.index == 'Australian_Shepherd_Dog']
    if not aus_data.empty:
        aus_row = aus_data.iloc[0]
        print(f"   Train: {aus_row['train']:.0f}, Val: {aus_row['val']:.0f}, Test: {aus_row['test']:.0f}")
        print(f"   Total: {aus_row['total']:.0f}")
        print(f"   Posizione ranking: {pivot_sorted.index.get_loc('Australian_Shepherd_Dog') + 1}/{len(pivot)}")
    else:
        print(f"   ❌ Australian_Shepherd_Dog NON trovato!")
    
    # Identifica razze problematiche
    print(f"\\n⚠️  RAZZE CON POSSIBILI PROBLEMI:")
    
    # Razze con troppo poche immagini
    low_count_breeds = pivot_sorted[pivot_sorted['total'] < 50]  # Soglia arbitraria
    if not low_count_breeds.empty:
        print(f"\\n   📉 Razze con < 50 immagini totali:")
        for breed, row in low_count_breeds.iterrows():
            print(f"      {breed}: {row['total']:.0f} immagini")
    
    # Razze con squilibrio train/val/test
    print(f"\\n   ⚖️  Controllo bilanciamento splits:")
    for breed, row in pivot_sorted.iterrows():
        train_pct = row['train'] / row['total'] * 100
        val_pct = row['val'] / row['total'] * 100  
        test_pct = row['test'] / row['total'] * 100
        
        # Expected: ~70/15/15
        if abs(train_pct - 70) > 10 or abs(val_pct - 15) > 5 or abs(test_pct - 15) > 5:
            print(f"      {breed}: Train {train_pct:.1f}%, Val {val_pct:.1f}%, Test {test_pct:.1f}%")
    
    # Raccomandazioni
    print(f"\\n" + "=" * 60)
    print(f"💡 RACCOMANDAZIONI")
    print(f"=" * 60)
    
    cv = total_counts.std() / total_counts.mean()
    if cv > 0.3:
        print(f"🚨 Dataset SBILANCIATO (CV={cv:.3f})")
        print(f"   💡 Usa WeightedRandomSampler durante training")
        print(f"   💡 Considera stratified sampling")
    else:
        print(f"✅ Dataset ragionevolmente bilanciato (CV={cv:.3f})")
    
    min_images = total_counts.min()
    if min_images < 30:
        print(f"\\n🚨 Alcune razze hanno TROPPO POCHE immagini (min={min_images})")
        print(f"   💡 Considera data augmentation più aggressivo")
        print(f"   💡 Potrebbe servire transfer learning")
    
    if train_counts.min() < 20:
        print(f"\\n🚨 Training set insufficiente per alcune razze (min train={train_counts.min()})")
        print(f"   💡 RACCOMANDAZIONE FORTE: Transfer Learning obbligatorio")
    
    return pivot_sorted

if __name__ == "__main__":
    dataset_stats = analyze_dataset()
