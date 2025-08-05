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
from utils.dataloader import create_dataloaders
from utils.config_helper import ConfigHelper

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
    """Test principale di validazione con test set separato"""
    print("🧪 Test di Validazione Progetto (Test Set Separato)")
    print("=" * 60)
    
    # Carica modello
    model, checkpoint = load_model()
    breed_names = get_breed_names()
    
    print(f"✅ Modello caricato")
    print(f"📊 Accuracy training: {checkpoint['train_acc']:.2f}%")
    print(f"📊 Accuracy validation: {checkpoint['val_acc']:.2f}%")
    
    # Carica dataloaders con split 70/15/15
    config = ConfigHelper('config.json')
    train_loader, val_loader, test_loader = create_dataloaders(config, max_breeds=10)
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_loader.dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    print(f"   Test: {len(test_loader.dataset)} samples")
    
    # Determina modalità di test
    if mode == 'australian':
        print(f"\n🔍 Testando Australian Shepherd su test set separato...")
        test_australian_on_separated_set(model, test_loader, breed_names)
    elif mode == 'sample':
        print(f"\n🔍 Testando 3 razze su test set separato...")
        test_sample_on_separated_set(model, test_loader, breed_names)
    else:  # all
        print(f"\n🔍 Testando tutte le razze su test set separato...")
        test_all_on_separated_set(model, test_loader, breed_names)
def test_australian_on_separated_set(model, test_loader, breed_names):
    """Test Australian Shepherd su test set separato"""
    model.eval()
    australian_idx = breed_names.index('Australian_Shepherd_Dog')
    
    correct = 0
    total = 0
    confidences = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            # Filtra solo Australian Shepherd
            for i in range(labels.size(0)):
                if labels[i].item() == australian_idx:
                    total += 1
                    confidence = probabilities[i][australian_idx].item() * 100
                    confidences.append(confidence)
                    
                    # Top prediction
                    _, predicted = torch.max(outputs[i], 0)
                    if predicted == australian_idx:
                        correct += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    
    print(f"🐕 Australian Shepherd Dog:")
    print(f"  ✅ Test set separato: {total} immagini")
    print(f"  📊 Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"  📊 Avg confidence: {avg_confidence:.1f}%")
    
    return accuracy, avg_confidence

def test_sample_on_separated_set(model, test_loader, breed_names):
    """Test 3 razze su test set separato"""
    model.eval()
    test_breeds = ['Australian_Shepherd_Dog', 'Afghan_hound', 'Bernese_mountain_dog']
    breed_indices = [breed_names.index(breed) for breed in test_breeds]
    
    breed_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                if label in breed_indices:
                    breed_name = breed_names[label]
                    breed_stats[breed_name]['total'] += 1
                    
                    confidence = probabilities[i][label].item() * 100
                    breed_stats[breed_name]['confidences'].append(confidence)
                    
                    # Top prediction
                    _, predicted = torch.max(outputs[i], 0)
                    if predicted == label:
                        breed_stats[breed_name]['correct'] += 1
    
    # Risultati
    print(f"\n📊 RISULTATI TEST SET SEPARATO")
    print(f"=" * 60)
    
    total_correct = 0
    total_images = 0
    
    for breed_name in test_breeds:
        stats = breed_stats[breed_name]
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        avg_confidence = np.mean(stats['confidences']) if stats['confidences'] else 0
        
        print(f"🐕 {breed_name}:")
        print(f"  📊 Accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
        print(f"  📊 Avg confidence: {avg_confidence:.1f}%")
        
        total_correct += stats['correct']
        total_images += stats['total']
    
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    print(f"\n🎯 Accuracy complessiva: {overall_accuracy:.1f}% ({total_correct}/{total_images})")
    
    # Raccomandazione
    print(f"\n💡 RACCOMANDAZIONE:")
    if overall_accuracy >= 70:
        print(f"  ✅ PROSEGUI CON IL PROGETTO (Accuracy {overall_accuracy:.1f}% >= 70%)")
    elif overall_accuracy >= 50:
        print(f"  ⚠️  PROSEGUI MA MIGLIORA IL MODELLO (Accuracy {overall_accuracy:.1f}%)")
    else:
        print(f"  ❌ NON PROCEDERE - MIGLIORA PRIMA IL MODELLO (Accuracy {overall_accuracy:.1f}% < 50%)")

def test_all_on_separated_set(model, test_loader, breed_names):
    """Test tutte le razze su test set separato"""
    model.eval()
    breed_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                breed_name = breed_names[label]
                breed_stats[breed_name]['total'] += 1
                
                confidence = probabilities[i][label].item() * 100
                breed_stats[breed_name]['confidences'].append(confidence)
                
                # Top prediction
                _, predicted = torch.max(outputs[i], 0)
                if predicted == label:
                    breed_stats[breed_name]['correct'] += 1
    
    # Risultati
    print(f"\n📊 RISULTATI TEST SET SEPARATO")
    print(f"=" * 60)
    
    total_correct = 0
    total_images = 0
    
    for breed_name in breed_names:
        stats = breed_stats[breed_name]
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            avg_confidence = np.mean(stats['confidences'])
            
            print(f"🐕 {breed_name}: {accuracy:.1f}% accuracy, {avg_confidence:.1f}% avg confidence")
            
            total_correct += stats['correct']
            total_images += stats['total']
    
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    print(f"\n🎯 Accuracy complessiva: {overall_accuracy:.1f}% ({total_correct}/{total_images})")
    
    # Raccomandazione
    print(f"\n💡 RACCOMANDAZIONE:")
    if overall_accuracy >= 70:
        print(f"  ✅ PROSEGUI CON IL PROGETTO (Accuracy {overall_accuracy:.1f}% >= 70%)")
    elif overall_accuracy >= 50:
        print(f"  ⚠️  PROSEGUI MA MIGLIORA IL MODELLO (Accuracy {overall_accuracy:.1f}%)")
    else:
        print(f"  ❌ NON PROCEDERE - MIGLIORA PRIMA IL MODELLO (Accuracy {overall_accuracy:.1f}% < 50%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test di validazione del progetto')
    parser.add_argument('--mode', choices=['all', 'australian', 'sample'], 
                       default='all', help='Modalità di test')
    
    args = parser.parse_args()
    test_validation(args.mode) 