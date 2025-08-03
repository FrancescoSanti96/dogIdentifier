#!/usr/bin/env python3
"""
Test unificato per validazione del progetto
"""

import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
import argparse
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.breed_classifier import create_breed_classifier

def load_model():
    """Carica il modello addestrato"""
    model_path = 'outputs/quick_test/quick_model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = create_breed_classifier(
        model_type='simple',
        num_classes=10,
        dropout_rate=0.3
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def get_breed_names():
    """Restituisce i nomi delle prime 10 razze"""
    return [
        'Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier',
        'Appenzeller', 'Australian_Shepherd_Dog', 'Australian_terrier', 'Bedlington_terrier',
        'Bernese_mountain_dog', 'Blenheim_spaniel'
    ]

def get_test_images(breed_name, num_images=3):
    """Ottiene immagini di test per una razza"""
    breed_dir = Path(f"data/breeds/{breed_name}")
    if not breed_dir.exists():
        return []
    
    images = list(breed_dir.glob('*.jpg'))[:num_images]
    return images

def test_single_image(model, image_path, breed_names, transform):
    """Testa una singola immagine"""
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Top 3 predizioni
            top_probs, top_indices = torch.topk(probabilities, 3)
        
        # Trova la razza corretta
        correct_breed = image_path.parent.name
        correct_idx = breed_names.index(correct_breed) if correct_breed in breed_names else -1
        
        # Confidence per la razza corretta
        correct_confidence = probabilities[0][correct_idx].item() * 100 if correct_idx >= 0 else 0
        
        # È nelle top 3?
        is_top3 = correct_idx in top_indices[0]
        
        return {
            'correct_breed': correct_breed,
            'correct_confidence': correct_confidence,
            'is_top3': is_top3,
            'top3': [(breed_names[idx], prob.item() * 100) for prob, idx in zip(top_probs[0], top_indices[0])]
        }
        
    except Exception as e:
        return {'error': str(e)}

def test_validation(mode='all'):
    """Test principale di validazione"""
    print("🧪 Test di Validazione Progetto")
    print("=" * 50)
    
    # Carica modello
    model, checkpoint = load_model()
    breed_names = get_breed_names()
    
    print(f"✅ Modello caricato")
    print(f"📊 Accuracy training: {checkpoint['train_acc']:.2f}%")
    print(f"📊 Accuracy validation: {checkpoint['val_acc']:.2f}%")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Determina razze da testare
    if mode == 'australian':
        test_breeds = ['Australian_Shepherd_Dog']
    elif mode == 'sample':
        test_breeds = ['Australian_Shepherd_Dog', 'Afghan_hound', 'Bernese_mountain_dog']
    else:  # all
        test_breeds = breed_names
    
    print(f"\n🔍 Testando {len(test_breeds)} razze in modalità '{mode}'...")
    
    # Statistiche per razza
    breed_stats = defaultdict(list)
    total_correct = 0
    total_images = 0
    
    for breed in test_breeds:
        print(f"\n🐕 Testando {breed}:")
        images = get_test_images(breed, 3)
        
        if not images:
            print(f"  ❌ Nessuna immagine trovata per {breed}")
            continue
        
        breed_correct = 0
        for i, img_path in enumerate(images):
            result = test_single_image(model, img_path, breed_names, transform)
            
            if 'error' in result:
                print(f"  ❌ Errore immagine {i+1}: {result['error']}")
                continue
            
            correct = result['is_top3']
            confidence = result['correct_confidence']
            
            if correct:
                breed_correct += 1
                total_correct += 1
                marker = "✅"
            else:
                marker = "❌"
            
            print(f"  {marker} Immagine {i+1}: {confidence:.1f}% confidence")
            
            breed_stats[breed].append({
                'confidence': confidence,
                'correct': correct
            })
        
        total_images += len(images)
        accuracy = (breed_correct / len(images)) * 100 if images else 0
        print(f"  📊 Accuracy {breed}: {accuracy:.1f}% ({breed_correct}/{len(images)})")
    
    # Risultati finali
    print(f"\n" + "="*50)
    print(f"📊 RISULTATI FINALI")
    print(f"="*50)
    
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    print(f"🎯 Accuracy complessiva: {overall_accuracy:.1f}% ({total_correct}/{total_images})")
    
    # Analisi per razza
    print(f"\n📋 Analisi per razza:")
    for breed in test_breeds:
        if breed in breed_stats:
            confidences = [r['confidence'] for r in breed_stats[breed]]
            avg_confidence = np.mean(confidences)
            correct_count = sum(1 for r in breed_stats[breed] if r['correct'])
            total_count = len(breed_stats[breed])
            accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            print(f"  {breed}: {accuracy:.1f}% accuracy, {avg_confidence:.1f}% avg confidence")
    
    # Raccomandazione
    print(f"\n💡 RACCOMANDAZIONE:")
    if overall_accuracy >= 70:
        print(f"  ✅ PROSEGUI CON IL PROGETTO (Accuracy {overall_accuracy:.1f}% >= 70%)")
    elif overall_accuracy >= 50:
        print(f"  ⚠️  PROSEGUI MA MIGLIORA IL DATASET (Accuracy {overall_accuracy:.1f}%)")
    else:
        print(f"  ❌ NON PROCEDERE - MIGLIORA PRIMA IL MODELLO (Accuracy {overall_accuracy:.1f}% < 50%)")
    
    # Focus su Australian Shepherd
    if 'Australian_Shepherd_Dog' in breed_stats:
        aus_stats = breed_stats['Australian_Shepherd_Dog']
        aus_accuracy = sum(1 for r in aus_stats if r['correct']) / len(aus_stats) * 100
        aus_avg_conf = np.mean([r['confidence'] for r in aus_stats])
        
        print(f"\n🐕 Australian Shepherd Dog:")
        print(f"  Accuracy: {aus_accuracy:.1f}%")
        print(f"  Avg confidence: {aus_avg_conf:.1f}%")
        
        if aus_accuracy < 50:
            print(f"  ⚠️  Australian Shepherd ha performance bassa - aggiungi più immagini!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test di validazione del progetto')
    parser.add_argument('--mode', choices=['all', 'australian', 'sample'], 
                       default='all', help='Modalità di test')
    
    args = parser.parse_args()
    test_validation(args.mode) 