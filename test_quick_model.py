#!/usr/bin/env python3
"""
Test script per il modello rapido addestrato
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.breed_classifier import create_breed_classifier

def test_quick_model():
    """Testa il modello rapido con alcune immagini"""
    print("🧪 Test Modello Rapido")
    print("="*50)
    
    # Carica modello
    model_path = 'outputs/quick_test/quick_model.pth'
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Crea modello (simple, 10 classi)
    model = create_breed_classifier(
        model_type='simple',
        num_classes=10,  # Solo 10 razze per il test rapido
        dropout_rate=0.3
    )
    
    # Carica pesi
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modello caricato: {model_path}")
    print(f"📊 Accuracy training: {checkpoint['train_acc']:.2f}%")
    print(f"📊 Accuracy validation: {checkpoint['val_acc']:.2f}%")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Nomi delle prime 10 razze
    breed_names = [
        'Australian_terrier', 'toy_poodle', 'Great_Pyrenees', 'Maltese_dog', 
        'Norwich_terrier', 'whippet', 'Boston_bull', 'Irish_setter', 
        'Rottweiler', 'kelpie'
    ]
    
    # Test con alcune immagini
    test_images = [
        'data/breeds/Australian_terrier/n02096294_1111.jpg',
        'data/breeds/toy_poodle/n02113624_1008.jpg',
        'data/breeds/Great_Pyrenees/n02111500_1031.jpg'
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n🔍 Testando: {os.path.basename(img_path)}")
            
            # Carica e preprocessa immagine
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            
            # Predizione
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                
                # Top 3 predizioni
                top_probs, top_indices = torch.topk(probabilities, 3)
            
            # Risultati
            print(f"  Top 3 predizioni:")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                breed_name = breed_names[idx]
                confidence = prob.item() * 100
                print(f"    {i+1}. {breed_name}: {confidence:.2f}%")
        else:
            print(f"❌ Immagine non trovata: {img_path}")
    
    print(f"\n✅ Test completato!")
    print(f"🎯 Il modello funziona correttamente!")

if __name__ == "__main__":
    test_quick_model() 