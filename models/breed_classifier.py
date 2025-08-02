#!/usr/bin/env python3
"""
Breed classifier model - Custom CNN architecture for dog breed classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BreedClassifier(nn.Module):
    """
    Custom CNN architecture for dog breed classification
    Built from scratch, no pre-trained models
    """
    
    def __init__(self, 
                 num_classes: int = 120, 
                 dropout_rate: float = 0.5, 
                 use_batch_norm: bool = True):
        """
        Initialize breed classifier
        
        Args:
            num_classes: Number of dog breeds to classify
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(BreedClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 3 -> 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 3: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Block 4: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # Block 5: 512 -> 512 channels
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleBreedClassifier(nn.Module):
    """
    Simplified CNN model for comparison experiments
    """
    
    def __init__(self, 
                 num_classes: int = 120, 
                 dropout_rate: float = 0.3, 
                 use_batch_norm: bool = True):
        """
        Initialize simple breed classifier
        
        Args:
            num_classes: Number of dog breeds to classify
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(SimpleBreedClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Simple feature extraction
        self.features = nn.Sequential(
            # Simple architecture: 3 conv layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_breed_classifier(model_type: str = 'full',
                           num_classes: int = 120,
                           dropout_rate: float = 0.5,
                           use_batch_norm: bool = True) -> nn.Module:
    """
    Factory function to create breed classifier
    
    Args:
        model_type: 'full' or 'simple'
        num_classes: Number of classes
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Initialized model
    """
    if model_type == 'full':
        model = BreedClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    elif model_type == 'simple':
        model = SimpleBreedClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary of model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Summary string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"Model Summary:\n"
    summary += f"  Total parameters: {total_params:,}\n"
    summary += f"  Trainable parameters: {trainable_params:,}\n"
    summary += f"  Model type: {model.__class__.__name__}\n"
    
    return summary


if __name__ == "__main__":
    # Test model creation
    print("🧪 Testing breed classifier models...")
    
    # Test full model
    full_model = create_breed_classifier('full', num_classes=120)
    print(get_model_summary(full_model))
    
    # Test simple model
    simple_model = create_breed_classifier('simple', num_classes=10)
    print(get_model_summary(simple_model))
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    full_output = full_model(input_tensor)
    simple_output = simple_model(input_tensor)
    
    print(f"✅ Full model output shape: {full_output.shape}")
    print(f"✅ Simple model output shape: {simple_output.shape}")
    
    print("✅ Breed classifier models test completed!") 