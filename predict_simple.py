#!/usr/bin/env python3
"""
Script semplice per predire razza da singola foto
Uso: python predict_simple.py foto.jpg outputs/models/breeds_10/best_model.pth
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.breed_classifier import create_breed_classifier

def predict_breed(image_path, model_path, top_k=3):
    """Predice razza da singola immagine"""
    
    # Carica modello
    print(f"📂 Caricando modello...")
    checkpoint = torch.load(model_path, map_location='cpu')
    num_classes = checkpoint['num_classes']
    breed_names = checkpoint.get('breed_names', [])
    
    # Auto-detect architettura
    state_dict = checkpoint['model_state_dict']
    
    # Rileva tipo di modello
    if any(k.startswith('layer1.') for k in state_dict.keys()):
        # Transfer Learning: ResNet18 backbone
        print(f"🔄 Rilevato: Transfer Learning (ResNet18)")
        model = create_breed_classifier(
            num_classes=num_classes, pretrained_backbone="resnet18", freeze_backbone=False)
    elif any('features.12' in k or 'features.15' in k for k in state_dict.keys()):
        # Full CNN: ha più layer (features.12, features.15, etc.)
        print(f"🏗️  Rilevato: Full CNN (134M parametri)")
        model = create_breed_classifier(model_type="full", num_classes=num_classes)
    else:
        # Simple CNN: meno layer
        print(f"⚡ Rilevato: Simple CNN (3.3M parametri)")
        model = create_breed_classifier(model_type="simple", num_classes=num_classes)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Preprocessa immagine
    print(f"🖼️  Processando immagine...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Predici
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    
    # Risultati
    print(f"\n🔮 PREDIZIONE PER: {image_path.split('/')[-1]}")
    print("=" * 50)
    
    for i, (prob, idx) in enumerate(zip(top_probs.squeeze(), top_indices.squeeze())):
        breed = breed_names[idx] if idx < len(breed_names) else f"Classe_{idx}"
        breed_clean = breed.replace('_', ' ').title()
        percentage = prob.item() * 100
        
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{emoji} {breed_clean:<25} {percentage:>6.2f}%")
    
    # Australian Shepherd check
    top_breed = breed_names[top_indices[0, 0]] if len(breed_names) > 0 else ""
    if 'Australian_Shepherd_Dog' in top_breed:
        print(f"\n⭐ AUSTRALIAN SHEPHERD RILEVATO! ⭐")

def main():
    if len(sys.argv) != 3:
        print("Uso: python predict_simple.py <immagine> <modello>")
        print("Esempio: python predict_simple.py cane.jpg outputs/models/breeds_10/best_model.pth")
        sys.exit(1)
    
    image_path, model_path = sys.argv[1], sys.argv[2]
    predict_breed(image_path, model_path)

if __name__ == "__main__":
    main()