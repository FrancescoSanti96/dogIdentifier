#!/usr/bin/env python3
"""
🐕 UNIVERSAL DOG CLASSIFIER
Smart prediction script che auto-detecta il tipo di modello e fa predizioni intelligenti

Usage:
    python predict.py <image> <model> [--top-k 3] [--binary-model path] [--threshold 0.5]

Examples:
    # Classificazione breed (auto-detecta)
    python predict.py dog.jpg outputs/models/breeds_10/best_model.pth
    
    # Con test binario automatico se trova Australian Shepherd
    python predict.py dog.jpg outputs/models/breeds_10/best_model.pth --binary-model outputs/my_dog/best_model.pth
    
    # Solo test binario
    python predict.py dog.jpg outputs/my_dog/best_model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.breed_classifier import create_breed_classifier

class UniversalDogClassifier:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Auto-detecta e carica il modello appropriato"""
        print(f"📂 Caricando modello: {os.path.basename(model_path)}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        num_classes = checkpoint.get('num_classes', 2)
        breed_names = checkpoint.get('breed_names', [])
        
        # Auto-detect architettura
        state_dict = checkpoint['model_state_dict']
        
        if num_classes == 2:
            # Modello binario
            print(f"🔍 Rilevato: MODELLO BINARIO (Maggie vs Altri)")
            model_type = "binary"
            model = create_breed_classifier(
                model_type="simple", 
                num_classes=2, 
                dropout_rate=0.5
            )
        elif any(k.startswith('layer1.') for k in state_dict.keys()):
            # Transfer Learning: ResNet18 backbone
            print(f"🔄 Rilevato: TRANSFER LEARNING ({num_classes} razze)")
            model_type = "transfer"
            model = create_breed_classifier(
                num_classes=num_classes, 
                pretrained_backbone="resnet18", 
                freeze_backbone=False
            )
        elif any('features.12' in k or 'features.15' in k for k in state_dict.keys()):
            # Full CNN
            print(f"🏗️  Rilevato: FULL CNN ({num_classes} razze)")
            model_type = "full"
            model = create_breed_classifier(
                model_type="full", 
                num_classes=num_classes
            )
        else:
            # Simple CNN
            print(f"⚡ Rilevato: SIMPLE CNN ({num_classes} razze)")
            model_type = "simple"
            model = create_breed_classifier(
                model_type="simple", 
                num_classes=num_classes
            )
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, model_type, num_classes, breed_names
    
    def predict_breeds(self, image_path, model_path, top_k=3):
        """Predizione razze multiple"""
        model, model_type, num_classes, breed_names = self.load_model(model_path)
        
        # Preprocessa immagine
        print(f"🖼️  Processando: {os.path.basename(image_path)}")
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"❌ Errore nel caricare l'immagine: {e}")
            return None, None
        
        # Predici
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probs, min(top_k, num_classes), dim=1)
        
        # Risultati
        print(f"\n🔮 CLASSIFICAZIONE RAZZE")
        print("=" * 50)
        
        results = []
        for i, (prob, idx) in enumerate(zip(top_probs.squeeze(), top_indices.squeeze())):
            breed = breed_names[idx] if idx < len(breed_names) else f"Classe_{idx}"
            breed_clean = breed.replace('_', ' ').title()
            percentage = prob.item() * 100
            
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{emoji} {breed_clean:<25} {percentage:>6.2f}%")
            
            results.append((breed, percentage))
        
        # Check per Australian Shepherd
        top_breed = breed_names[top_indices[0, 0]] if len(breed_names) > 0 else ""
        is_australian = 'Australian_Shepherd_Dog' in top_breed or 'australian' in top_breed.lower()
        
        if is_australian:
            print(f"\n⭐ AUSTRALIAN SHEPHERD RILEVATO! ⭐")
            
        return results, is_australian
    
    def predict_binary(self, image_path, model_path):
        """Predizione binaria Maggie vs Altri"""
        model, model_type, num_classes, breed_names = self.load_model(model_path)
        
        if num_classes != 2:
            print(f"⚠️ ATTENZIONE: Modello non binario ({num_classes} classi)")
            return None
        
        # Preprocessa immagine
        print(f"🖼️  Processando: {os.path.basename(image_path)}")
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"❌ Errore nel caricare l'immagine: {e}")
            return None
        
        # Predici
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            other_prob = probs[0, 0].item()   # Classe 0 = Altri
            maggie_prob = probs[0, 1].item()  # Classe 1 = Maggie
        
        # Risultati
        print(f"\n🔮 CLASSIFICAZIONE BINARIA")
        print("=" * 40)
        
        predicted_class = probs[0].argmax().item()
        is_maggie = (predicted_class == 1)
        confidence = probs[0].max().item() * 100
        
        if is_maggie:
            print(f"🏆 RISULTATO: È MAGGIE! 🐕")
            confidence_level = "ALTA" if confidence >= 80 else "MEDIA" if confidence >= 60 else "BASSA"
            status_emoji = "✅" if confidence >= 80 else "⚠️" if confidence >= 60 else "❓"
        else:
            print(f"🔍 RISULTATO: NON è Maggie (altro cane) 🐶")
            confidence_level = "ALTA" if confidence >= 80 else "MEDIA" if confidence >= 60 else "BASSA"
            status_emoji = "✅" if confidence >= 80 else "⚠️" if confidence >= 60 else "❓"
        
        print(f"🎯 Confidenza: {confidence:.1f}% ({confidence_level})")
        print(f"📊 Maggie: {maggie_prob*100:.1f}% | Altri: {other_prob*100:.1f}%")
        print(f"{status_emoji} CONFIDENZA {confidence_level}")
        
        return {
            'is_maggie': is_maggie,
            'confidence': confidence,
            'maggie_prob': maggie_prob,
            'other_prob': other_prob,
            'predicted_class': predicted_class
        }

def main():
    parser = argparse.ArgumentParser(description="🐕 Universal Dog Classifier")
    parser.add_argument("image", help="Path dell'immagine da classificare")
    parser.add_argument("model", help="Path del modello principale")
    parser.add_argument("--top-k", type=int, default=3, help="Top K predizioni per breed classification")
    parser.add_argument("--binary-model", help="Path del modello binario per test Maggie (opzionale)")
    parser.add_argument("--binary-only", action="store_true", help="Esegui solo test binario")
    parser.add_argument("--threshold", type=float, default=0.3, help="Soglia per trigger test binario automatico")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Immagine non trovata: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Modello non trovato: {args.model}")
        sys.exit(1)
    
    classifier = UniversalDogClassifier()
    
    print("🐕 UNIVERSAL DOG CLASSIFIER")
    print("=" * 60)
    print(f"📸 Immagine: {args.image}")
    print(f"🧠 Modello: {args.model}")
    
    # Determina tipo di analisi
    if args.binary_only:
        # Solo test binario
        print(f"\n🎯 MODALITÀ: Solo test binario")
        result = classifier.predict_binary(args.image, args.model)
        
    else:
        # Test breed classification
        print(f"\n🎯 MODALITÀ: Classificazione razze")
        breed_results, is_australian = classifier.predict_breeds(args.image, args.model, args.top_k)
        
        # Test binario automatico se Australian Shepherd e modello disponibile
        if is_australian and args.binary_model and os.path.exists(args.binary_model):
            print(f"\n🔄 TRIGGER AUTOMATICO: Test binario Maggie")
            print(f"🧠 Modello binario: {args.binary_model}")
            binary_result = classifier.predict_binary(args.image, args.binary_model)
            
            if binary_result and binary_result['is_maggie']:
                print(f"\n🎉 VERDETTO FINALE: Australian Shepherd + È MAGGIE!")
            elif binary_result:
                print(f"\n🤔 VERDETTO FINALE: Australian Shepherd ma NON è Maggie")
        
        elif is_australian and args.binary_model:
            print(f"\n⚠️ Modello binario specificato ma non trovato: {args.binary_model}")
        
        elif is_australian:
            print(f"\n💡 SUGGERIMENTO: Aggiungi --binary-model per test Maggie automatico")
    
    print(f"\n✅ Analisi completata!")

if __name__ == "__main__":
    main()
