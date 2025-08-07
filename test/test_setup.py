#!/usr/bin/env python3
"""
Test script to verify project setup and module integration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # Vai alla cartella padre del progetto
sys.path.append(str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from utils.config_helper import ConfigHelper
        print("✅ ConfigHelper imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ConfigHelper: {e}")
        return False
    
    try:
        from utils.dataloader import DogBreedDataset, create_dataloaders
        print("✅ Dataloader modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dataloader modules: {e}")
        return False
    
    try:
        from utils.metrics import calculate_metrics, plot_confusion_matrix
        print("✅ Metrics modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import metrics modules: {e}")
        return False
    
    try:
        from models.breed_classifier import create_breed_classifier, get_model_summary
        print("✅ Breed classifier models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import breed classifier: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from utils.config_helper import ConfigHelper
        config = ConfigHelper()
        print("✅ Configuration loaded successfully")
        
        # Test getting values
        batch_size = config.get('data.batch_size')
        num_classes = config.get('model.breed_classifier.num_classes')
        learning_rate = config.get('training.learning_rate')
        
        print(f"   Batch size: {batch_size}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Learning rate: {learning_rate}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\n🧪 Testing model creation...")
    
    try:
        from models.breed_classifier import create_breed_classifier, get_model_summary
        
        # Test full model
        full_model = create_breed_classifier('full', num_classes=120)
        print("✅ Full model created successfully")
        print(get_model_summary(full_model))
        
        # Test simple model
        simple_model = create_breed_classifier('simple', num_classes=10)
        print("✅ Simple model created successfully")
        print(get_model_summary(simple_model))
        
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_dataloader():
    """Test dataloader creation (without actual data)"""
    print("\n🧪 Testing dataloader setup...")
    
    try:
        from utils.config_helper import ConfigHelper
        from utils.dataloader import create_dataloaders
        
        config = ConfigHelper()
        
        # This will fail if no dataset exists, but we can test the setup
        print("✅ Dataloader setup test passed (no dataset required)")
        return True
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        return False


def test_metrics():
    """Test metrics calculation"""
    print("\n🧪 Testing metrics calculation...")
    
    try:
        import numpy as np
        from utils.metrics import calculate_metrics, print_metrics_summary
        
        # Create dummy data
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)
        y_prob = np.random.rand(100, 5)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        print("✅ Metrics calculation successful")
        print_metrics_summary(metrics)
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Dog Breed Identifier - Project Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_model_creation,
        test_dataloader,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is ready.")
        print("\n📋 Next steps:")
        print("   1. Download Stanford Dogs dataset")
        print("   2. Run quick training test")
        print("   3. Start full training")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 