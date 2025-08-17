#!/usr/bin/env python3
"""
Script per confrontare le architetture senza training completo.
Mostra differenze in parametri, FLOPs e complessità.
"""

import torch
from models.breed_classifier import create_breed_classifier

def count_parameters(model):
    """Conta parametri totali e trainable."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def analyze_model_complexity():
    """Analizza complessità delle 3 architetture."""
    
    print("🔬 ANALISI COMPLESSITÀ ARCHITETTURE")
    print("=" * 50)
    
    models = {
        "SimpleBreedClassifier": {
            "model": create_breed_classifier(30, architecture="simple"),
            "description": "CNN leggera (3 blocchi conv)"
        },
        "BreedClassifier (Full)": {
            "model": create_breed_classifier(30, architecture="full"), 
            "description": "CNN profonda VGG-like (5 blocchi conv)"
        },
        "ResNet18 Transfer Learning": {
            "model": create_breed_classifier(30, pretrained_backbone="resnet18", freeze_backbone=True),
            "description": "ResNet18 pre-addestrato (frozen backbone)"
        }
    }
    
    results = []
    
    for name, info in models.items():
        model = info["model"]
        total_params, trainable_params = count_parameters(model)
        
        # Test forward pass per verificare funzionamento
        dummy_input = torch.randn(1, 3, 224, 224)
        try:
            with torch.no_grad():
                output = model(dummy_input)
            forward_ok = "✅"
        except Exception as e:
            forward_ok = f"❌ {str(e)[:50]}"
        
        results.append({
            "name": name,
            "description": info["description"],
            "total_params": total_params,
            "trainable_params": trainable_params,
            "forward_test": forward_ok,
            "efficiency": trainable_params / total_params
        })
        
        print(f"\n📊 {name}")
        print(f"   {info['description']}")
        print(f"   Parametri totali: {total_params:,}")
        print(f"   Parametri trainable: {trainable_params:,}")
        print(f"   Efficienza: {trainable_params/total_params:.1%}")
        print(f"   Forward pass: {forward_ok}")
    
    # Tabella comparativa
    print("\n" + "=" * 80)
    print("📋 TABELLA COMPARATIVA")
    print("=" * 80)
    
    print(f"{'Architettura':<25} {'Parametri Tot':<15} {'Trainable':<15} {'Ratio':<10}")
    print("-" * 80)
    
    for r in results:
        ratio = f"{r['trainable_params'] / results[0]['trainable_params']:.1f}x"
        print(f"{r['name']:<25} {r['total_params']:>12,} {r['trainable_params']:>12,} {ratio:>8}")
    
    # Performance teorica attesa
    print(f"\n🎯 PERFORMANCE TEORICA ATTESA (30 razze):")
    print(f"   SimpleBreedClassifier:     18.73% (TESTATO)")
    print(f"   BreedClassifier (Full):    ~25-35% (stima)")
    print(f"   ResNet18 Transfer Learning: 78.0% (TESTATO)")
    
    return results

if __name__ == "__main__":
    analyze_model_complexity()

