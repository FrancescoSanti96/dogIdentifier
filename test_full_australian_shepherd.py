#!/usr/bin/env python3
"""
Test del modello completo su Australian Shepherd Dog
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataloader import create_dataloaders_from_splits
from models.breed_classifier import create_breed_classifier

def test_full_australian_shepherd():
    """Test specifico su Australian Shepherd con modello completo"""
    print("🐕 AUSTRALIAN SHEPHERD TEST - Full Model (121 Razze)")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'outputs/full_training/best_full_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato: {model_path}")
        print("   Esegui prima: python full_train.py")
        return
    
    # Load model
    print("📥 Caricando modello completo...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Get model info
    num_classes = checkpoint['num_classes']
    breed_names = checkpoint['breed_names']
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"Modello info:")
    print(f"  Classi totali: {num_classes}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    # Check Australian Shepherd
    if 'Australian_Shepherd_Dog' not in breed_names:
        print(f"❌ Australian_Shepherd_Dog non trovato nelle classi del modello!")
        return
    
    aus_class_idx = breed_names.index('Australian_Shepherd_Dog')
    print(f"  ✅ Australian_Shepherd_Dog trovato (classe {aus_class_idx})")
    
    # Create model
    model = create_breed_classifier(
        model_type='full',
        num_classes=num_classes,
        dropout_rate=checkpoint.get('config', {}).get('model', {}).get('breed_classifier', {}).get('dropout_rate', 0.5)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("\\n📊 Caricando dataset di test...")
    _, _, test_loader = create_dataloaders_from_splits(
        splits_dir='data/full_splits',
        batch_size=32,
        num_workers=4,
        image_size=(224, 224)
    )
    
    print(f"Test samples totali: {len(test_loader.dataset):,}")
    
    # Test Australian Shepherd specifically
    print("\\n🎯 Testing Australian Shepherd Dog...")
    
    aus_correct = 0
    aus_total = 0
    aus_confidences = []
    aus_predictions = []
    
    all_correct = 0
    all_total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            # Overall accuracy
            all_total += target.size(0)
            all_correct += predicted.eq(target).sum().item()
            
            # Australian Shepherd specific analysis
            aus_mask = (target == aus_class_idx)
            if aus_mask.any():
                aus_indices = aus_mask.nonzero().squeeze()
                if aus_indices.dim() == 0:
                    aus_indices = aus_indices.unsqueeze(0)
                
                for idx in aus_indices:
                    aus_total += 1
                    true_label = target[idx].item()
                    pred_label = predicted[idx].item()
                    confidence = probabilities[idx, aus_class_idx].item() * 100
                    
                    aus_confidences.append(confidence)
                    aus_predictions.append(pred_label)
                    
                    if pred_label == true_label:
                        aus_correct += 1
            
            # Update progress
            current_acc = 100. * all_correct / all_total
            pbar.set_postfix({'Overall Acc': f'{current_acc:.2f}%'})
    
    # Calculate results
    overall_accuracy = 100. * all_correct / all_total
    aus_accuracy = (100. * aus_correct / aus_total) if aus_total > 0 else 0.0
    avg_confidence = sum(aus_confidences) / len(aus_confidences) if aus_confidences else 0.0
    
    print(f"\\n" + "=" * 60)
    print(f"📊 RISULTATI TEST")
    print(f"=" * 60)
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}% ({all_correct}/{all_total})")
    print(f"\\n🐕 Australian Shepherd Dog Results:")
    print(f"   Test images: {aus_total}")
    print(f"   Correct predictions: {aus_correct}")
    print(f"   Accuracy: {aus_accuracy:.2f}%")
    print(f"   Average confidence: {avg_confidence:.1f}%")
    
    if aus_total > 0:
        # Analyze errors
        errors = aus_total - aus_correct
        print(f"   Errors: {errors}")
        
        if errors > 0:
            print(f"\\n❌ Error Analysis:")
            error_classes = {}
            for i, pred_idx in enumerate(aus_predictions):
                if pred_idx != aus_class_idx:
                    error_breed = breed_names[pred_idx]
                    error_classes[error_breed] = error_classes.get(error_breed, 0) + 1
            
            for breed, count in sorted(error_classes.items(), key=lambda x: x[1], reverse=True):
                print(f"   • Confused with {breed}: {count} times")
        
        # Confidence analysis
        high_conf = len([c for c in aus_confidences if c > 80])
        med_conf = len([c for c in aus_confidences if 50 <= c <= 80])
        low_conf = len([c for c in aus_confidences if c < 50])
        
        print(f"\\n📊 Confidence Distribution:")
        print(f"   High confidence (>80%): {high_conf} images ({high_conf/aus_total*100:.1f}%)")
        print(f"   Medium confidence (50-80%): {med_conf} images ({med_conf/aus_total*100:.1f}%)")
        print(f"   Low confidence (<50%): {low_conf} images ({low_conf/aus_total*100:.1f}%)")
        
        # Performance assessment
        print(f"\\n🎯 PERFORMANCE ASSESSMENT:")
        if aus_accuracy >= 80:
            print("   ✅ EXCELLENT: Australian Shepherd recognition ≥80%")
        elif aus_accuracy >= 70:
            print("   ✅ GOOD: Australian Shepherd recognition ≥70%")
        elif aus_accuracy >= 60:
            print("   ⚠️  FAIR: Australian Shepherd recognition ≥60%")
        else:
            print("   ❌ POOR: Australian Shepherd recognition <60%")
        
        # Compare with training results
        if 'history' in checkpoint:
            final_train_acc = checkpoint.get('train_acc', 0.0)
            final_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"\\n📈 Training Comparison:")
            print(f"   Training accuracy: {final_train_acc:.2f}%")
            print(f"   Validation accuracy: {final_val_acc:.2f}%")
            print(f"   Test accuracy: {overall_accuracy:.2f}%")
            print(f"   Australian Shepherd: {aus_accuracy:.2f}%")
        
        # Save detailed results
        results = {
            'model_info': {
                'total_classes': num_classes,
                'best_val_acc': best_val_acc,
                'model_path': model_path
            },
            'test_results': {
                'overall_accuracy': overall_accuracy,
                'total_test_samples': all_total,
                'correct_predictions': all_correct
            },
            'australian_shepherd': {
                'test_images': aus_total,
                'correct_predictions': aus_correct,
                'accuracy': aus_accuracy,
                'average_confidence': avg_confidence,
                'confidences': aus_confidences,
                'error_analysis': error_classes if 'error_classes' in locals() else {},
                'confidence_distribution': {
                    'high_confidence': high_conf,
                    'medium_confidence': med_conf,
                    'low_confidence': low_conf
                }
            }
        }
        
        os.makedirs('outputs/full_training', exist_ok=True)
        with open('outputs/full_training/australian_shepherd_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n💾 Detailed results saved: outputs/full_training/australian_shepherd_test_results.json")
    
    print(f"\\n✅ Test completed!")

if __name__ == "__main__":
    test_full_australian_shepherd()
