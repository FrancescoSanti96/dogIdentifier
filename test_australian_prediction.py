#!/usr/bin/env python3
"""
Test delle predizioni per Australian Shepherd Dog
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models.breed_classifier import SimpleBreedClassifier

def test_australian_shepherd():
    """Test delle predizioni per Australian Shepherd"""
    
    # Carica il modello
    device = torch.device('cpu')
    model = SimpleBreedClassifier(num_classes=5)
    
    # Carica i pesi salvati dal training corretto
    checkpoint_path = 'outputs/quick_splits/quick_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Modello non trovato: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modello caricato da: {checkpoint_path}")
    print(f"🔧 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Classi del quick test (dall'ordine in cui sono stati caricati)
    class_names = ['Australian_Shepherd_Dog', 'Japanese_spaniel', 'Lhasa', 'Norwich_terrier', 'miniature_pinscher']
    
    print(f"🎯 Classes: {class_names}")
    
    # Transform per le immagini
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Usa il dataset di TEST già preparato con prepare_dataset.py
    aus_path = 'data/quick_splits/test/Australian_Shepherd_Dog'
    
    if not os.path.exists(aus_path):
        print(f"❌ Dataset di test non trovato: {aus_path}")
        print("💡 Esegui 'python prepare_dataset.py' per creare i dataset splits")
        return
    
    aus_images = [f for f in os.listdir(aus_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n🔍 Test su immagini di TEST (mai viste durante training):")
    print(f"   📁 Dataset di test: {aus_path}")
    print(f"   🎯 Testando: {len(aus_images)} immagini di test")
    
    correct_predictions = 0
    aus_class_idx = class_names.index('Australian_Shepherd_Dog')
    
    with torch.no_grad():
        for img_name in aus_images:
            img_path = os.path.join(aus_path, img_name)
            
            try:
                # Carica e preprocessa l'immagine
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)
                
                # Predizione
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(outputs, dim=1).item()
                predicted_class = class_names[predicted_class_idx]
                confidence = probabilities[0][predicted_class_idx].item() * 100
                
                # Verifica se è corretto
                is_correct = predicted_class_idx == aus_class_idx
                if is_correct:
                    correct_predictions += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {img_name[:30]:<30} -> {predicted_class:<25} ({confidence:.1f}%)")
                
                # Mostra top 3 predizioni
                top3_probs, top3_indices = torch.topk(probabilities[0], 3)
                print(f"    Top 3: ", end="")
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    class_name = class_names[idx.item()]
                    print(f"{class_name} ({prob.item()*100:.1f}%)", end="")
                    if i < 2:
                        print(", ", end="")
                print()
                
            except Exception as e:
                print(f"❌ Errore con {img_name}: {e}")
    
    accuracy = correct_predictions / len(aus_images) * 100
    print(f"\n🎯 Accuracy Australian Shepherd: {correct_predictions}/{len(aus_images)} = {accuracy:.1f}%")
    
    if accuracy >= 60:
        print("✅ Buona performance su Australian Shepherd!")
    elif accuracy >= 40:
        print("⚠️ Performance discreta, ma migliorabile")
    else:
        print("❌ Performance bassa, necessario più training")

if __name__ == "__main__":
    test_australian_shepherd()
