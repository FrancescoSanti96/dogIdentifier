#!/usr/bin/env python3
"""
Test Enhanced Model - Specifically for Australian Shepherd Recognition
"""

import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from torchvision import transforms
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.breed_classifier import SimpleBreedClassifier
from utils.dataloader import create_dataloaders_from_splits

def load_enhanced_model():
    """Load the enhanced trained model"""
    model_path = '../../outputs/quick_enhanced/best_model_enhanced.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Enhanced model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Use the number of classes from the checkpoint
    num_classes = checkpoint.get('num_classes', 5)
    
    model = SimpleBreedClassifier(
        num_classes=num_classes,
        dropout_rate=0.5  # Enhanced model uses 0.5 dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def get_breed_names():
    """Return the names of the 5 breeds in quick dataset"""
    return [
        'Australian_Shepherd_Dog', 'Japanese_spaniel', 'Lhasa', 
        'Norwich_terrier', 'miniature_pinscher'
    ]

def test_enhanced_model_australian():
    """Test enhanced model specifically on Australian Shepherd recognition"""
    print("🧪 Enhanced Model Test - Australian Shepherd Recognition")
    print("=" * 60)
    
    # Load enhanced model
    print("Loading enhanced model...")
    model, checkpoint = load_enhanced_model()
    
    print(f"✅ Enhanced model loaded")
    print(f"📊 Model validation accuracy: {checkpoint.get('final_val_acc', 'N/A'):.2f}%")
    print(f"📊 Model test accuracy: {checkpoint.get('final_test_acc', 'N/A'):.2f}%")
    
    # Load test data
    test_dir = "../../data/quick_splits/test"
    australian_test_dir = os.path.join(test_dir, "Australian_Shepherd_Dog")
    
    if not os.path.exists(australian_test_dir):
        print(f"❌ Australian Shepherd test directory not found: {australian_test_dir}")
        return
    
    # Get test images
    test_images = []
    for img_file in os.listdir(australian_test_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(australian_test_dir, img_file))
    
    print(f"\\n🔍 Test su immagini di TEST (mai viste durante training):")
    print(f"   📁 Dataset di test: {australian_test_dir}")
    print(f"   🎯 Testando: {len(test_images)} immagini di test")
    
    # Preprocessing (same as validation/test)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test each image
    breed_names = get_breed_names()
    australian_idx = breed_names.index('Australian_Shepherd_Dog')
    
    correct = 0
    predictions = []
    confidences = []
    
    print(f"\\n📋 Detailed Results:")
    print("-" * 60)
    
    with torch.no_grad():
        for i, img_path in enumerate(test_images, 1):
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # Predict
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_pct = confidence.item() * 100
            
            is_correct = predicted_idx == australian_idx
            correct += is_correct
            
            predictions.append(predicted_idx)
            confidences.append(confidence_pct)
            
            # Display result
            predicted_breed = breed_names[predicted_idx]
            status = "✅" if is_correct else "❌"
            
            print(f"{i:2d}. {os.path.basename(img_path):<25} → {predicted_breed:<20} ({confidence_pct:5.1f}%) {status}")
            
            if not is_correct:
                print(f"    Expected: Australian_Shepherd_Dog, Got: {predicted_breed}")
    
    # Calculate metrics
    accuracy = correct / len(test_images) * 100
    avg_confidence = np.mean(confidences)
    
    print("\\n" + "=" * 60)
    print("📊 ENHANCED MODEL AUSTRALIAN SHEPHERD RESULTS")
    print("=" * 60)
    print(f"🎯 Accuracy Australian Shepherd: {correct}/{len(test_images)} = {accuracy:.1f}%")
    print(f"🎯 Average Confidence: {avg_confidence:.1f}%")
    
    # Compare with baseline
    print(f"\\n📈 COMPARISON WITH BASELINE:")
    print(f"• Baseline Australian Shepherd: 60.9% (comprehensive) / 63.6% (focused)")
    print(f"• Enhanced Australian Shepherd: {accuracy:.1f}%")
    
    if accuracy >= 63.6:
        print(f"✅ Enhanced model MATCHES or BEATS baseline Australian Shepherd recognition!")
    elif accuracy >= 60.0:
        print(f"⚡ Enhanced model performance comparable to baseline")
    else:
        print(f"❌ Enhanced model underperforms on Australian Shepherd recognition")
    
    # Error analysis
    if correct < len(test_images):
        print(f"\\n🔍 ERROR ANALYSIS:")
        error_count = defaultdict(int)
        for i, pred_idx in enumerate(predictions):
            if pred_idx != australian_idx:
                error_breed = breed_names[pred_idx]
                error_count[error_breed] += 1
        
        print("Confusion with other breeds:")
        for breed, count in error_count.items():
            print(f"  • {breed}: {count} errors")
    
    print(f"\\n📁 Enhanced model path: outputs/quick_enhanced/best_model_enhanced.pth")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(test_images),
        'avg_confidence': avg_confidence,
        'model_type': 'Enhanced'
    }

def test_enhanced_full_dataset():
    """Test enhanced model on full test dataset (all 5 breeds)"""
    print("\\n🧪 Enhanced Model Test - Full Dataset (All 5 Breeds)")
    print("=" * 60)
    
    # Load model
    model, checkpoint = load_enhanced_model()
    
    # Create test dataloader
    try:
        _, _, test_loader = create_dataloaders_from_splits(
            "../../data/quick_splits", 
            batch_size=32, 
            num_workers=0
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return
    
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    correct = 0
    total = 0
    breed_correct = defaultdict(int)
    breed_total = defaultdict(int)
    breed_names = get_breed_names()
    
    print("Testing enhanced model on full dataset...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-breed accuracy
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                breed_total[true_label] += 1
                if true_label == pred_label:
                    breed_correct[true_label] += 1
    
    overall_accuracy = 100.0 * correct / total
    
    print(f"\\n📊 ENHANCED MODEL FULL DATASET RESULTS:")
    print(f"🎯 Overall Accuracy: {overall_accuracy:.1f}% ({correct}/{total})")
    
    print(f"\\n📊 Per-breed accuracy:")
    for i, breed_name in enumerate(breed_names):
        if breed_total[i] > 0:
            breed_acc = 100.0 * breed_correct[i] / breed_total[i]
            print(f"🐕 {breed_name}: {breed_acc:.1f}% ({breed_correct[i]}/{breed_total[i]})")
    
    return {
        'overall_accuracy': overall_accuracy,
        'breed_accuracies': {breed_names[i]: 100.0 * breed_correct[i] / breed_total[i] 
                           for i in range(len(breed_names)) if breed_total[i] > 0}
    }

if __name__ == "__main__":
    # Test Australian Shepherd specifically
    australian_results = test_enhanced_model_australian()
    
    # Test full dataset
    full_results = test_enhanced_full_dataset()
    
    print(f"\\n{'='*60}")
    print("ENHANCED MODEL COMPLETE EVALUATION")
    print(f"{'='*60}")
    print(f"✅ Australian Shepherd Accuracy: {australian_results['accuracy']:.1f}%")
    print(f"✅ Overall Test Accuracy: {full_results['overall_accuracy']:.1f}%")
    print(f"✅ Enhanced model evaluation complete!")
