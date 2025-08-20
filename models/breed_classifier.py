#!/usr/bin/env python3
"""
Modelli classificatori di razze - Architetture CNN personalizzate per classificazione razze canine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torchvision import models


class BreedClassifier(nn.Module):
    """
    Architettura CNN personalizzata per classificazione razze canine
    Costruita da zero, senza modelli pre-addestrati

    Architettura VGG-like personalizzata con 134M parametri:
    - 5 blocchi convoluzionali con feature maps crescenti (64→128→256→512→512)
    - Batch normalization per stabilità training
    - Dropout 2D per regolarizzazione convoluzionale
    - 3 layer fully connected per classificazione finale
    - Adaptive average pooling per flessibilità input size
    """

    def __init__(
        self,
        num_classes: int = 120,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """
        Inizializza il classificatore di razze

        Args:
            num_classes: Numero di razze canine da classificare
            dropout_rate: Tasso di dropout per la regolarizzazione
            use_batch_norm: Se utilizzare batch normalization
        """
        super(BreedClassifier, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Feature extraction layers - Architettura VGG-like con miglioramenti moderni
        self.features = nn.Sequential(
            # Blocco 1: 3 -> 64 canali (224x224 -> 112x112)
            # Primo blocco: estrae feature di basso livello (edges, textures)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Mantiene spatial size
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),  # Normalizzazione
            nn.ReLU(inplace=True),  # Attivazione non-lineare
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Doppia convoluzione
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling 2x
            nn.Dropout2d(dropout_rate * 0.5),  # Dropout spaziale ridotto
            # Blocco 2: 64 -> 128 canali (112x112 -> 56x56)
            # Secondo blocco: feature di medio livello (patterns, shapes)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsampling 2x
            nn.Dropout2d(dropout_rate * 0.5),
            # Blocco 3: 128 -> 256 canali
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
            # Blocco 4: 256 -> 512 canali
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
            # Blocco 5: 512 -> 512 canali
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

        # Classifier layers - Classificatore finale con regolarizzazione
        self.classifier = nn.Sequential(
            # Adaptive pooling: converte qualsiasi size in 7x7 (flessibilità input)
            nn.AdaptiveAvgPool2d((7, 7)),  # 512 x 7 x 7 = 25,088 features
            nn.Flatten(),  # Converte in vettore 1D
            # Primo FC layer: riduzione dimensionalità con regolarizzazione
            nn.Linear(512 * 7 * 7, 4096),  # 25,088 -> 4,096 features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Dropout per prevenire overfitting
            # Secondo FC layer: ulteriore elaborazione features
            nn.Linear(4096, 4096),  # Mantiene 4,096 features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # Output layer: classificazione finale (no activation, gestita da loss)
            nn.Linear(4096, num_classes),  # 4,096 -> num_classes logits
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Inizializza i pesi del modello usando le migliori pratiche

        Utilizza inizializzazioni specifiche per ogni tipo di layer:
        - Conv2D: Kaiming normal (ottimale per ReLU)
        - BatchNorm: weight=1, bias=0 (standard)
        - Linear: distribuzione normale con std piccola
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Inizializzazione Kaiming per layer convoluzionali con ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Batch normalization: weight=1 (no scaling), bias=0 (no shift)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Layer lineari: distribuzione normale piccola
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passaggio in avanti

        Args:
            x: Tensore di input di forma (batch_size, 3, height, width)

        Returns:
            Tensore di output di forma (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleBreedClassifier(nn.Module):
    """
    Modello CNN semplificato per esperimenti di confronto
    """

    def __init__(
        self,
        num_classes: int = 120,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Inizializza il classificatore semplificato

        Args:
            num_classes: Numero di razze canine da classificare
            dropout_rate: Tasso di dropout per la regolarizzazione
            use_batch_norm: Se utilizzare batch normalization
        """
        super(SimpleBreedClassifier, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Estrazione feature semplice
        self.features = nn.Sequential(
            # Architettura semplice: 3 layer convoluzionali
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

        # Classificatore semplice
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializza i pesi del modello"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passaggio in avanti

        Args:
            x: Tensore di input di forma (batch_size, 3, height, width)

        Returns:
            Tensore di output di forma (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_breed_classifier(
    model_type: str = "full",
    num_classes: int = 120,
    dropout_rate: float = 0.5,
    use_batch_norm: bool = True,
    pretrained_backbone: Optional[str] = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Funzione factory per creare classificatori di razze

    Args:
        model_type: 'full' o 'simple'
        num_classes: Numero di classi
        dropout_rate: Tasso di dropout
        use_batch_norm: Se utilizzare batch normalization
        pretrained_backbone: Backbone pre-addestrato (es. 'resnet18')
        freeze_backbone: Se congelare il backbone

    Returns:
        Modello inizializzato
    """
    if pretrained_backbone:
        # Transfer Learning Path: utilizza modello pre-addestrato su ImageNet
        # Vantaggi: convergenza più veloce, meno dati richiesti, feature generiche già apprese
        if pretrained_backbone.lower() == "resnet18":
            # Carica ResNet18 con pesi ImageNet (11M parametri backbone)
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = backbone.fc.in_features  # 512 feature ResNet18

            # Sostituisci classificatore finale: da 1000 classi ImageNet → num_classes razze
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),  # Regolarizzazione finale
                nn.Linear(in_features, num_classes),  # 512 → num_classes
            )
            model = backbone

            # Feature Extraction vs Fine-tuning
            if freeze_backbone:
                # Freeze backbone: solo classificatore finale trainable (~61K parametri)
                for name, param in model.named_parameters():
                    if not name.startswith(
                        "fc."
                    ):  # Mantieni solo testa finale trainable
                        param.requires_grad = False
        else:
            raise ValueError(
                f"Backbone pre-addestrato non supportato: {pretrained_backbone}"
            )
    elif model_type == "full":
        # From Scratch Path - Full: architettura CNN personalizzata completa
        # 134M parametri, VGG-like, tutti i layer trainable da zero
        model = BreedClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
    elif model_type == "simple":
        # From Scratch Path - Simple: architettura CNN semplificata
        # 3.3M parametri, per test rapidi e confronti
        model = SimpleBreedClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
    else:
        raise ValueError(f"Tipo di modello sconosciuto: {model_type}")

    return model


def get_model_summary(model: nn.Module) -> str:
    """
    Ottieni un riassunto dei parametri del modello

    Args:
        model: Modello PyTorch

    Returns:
        Stringa riassuntiva
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = f"Riassunto Modello:\n"
    summary += f"  Parametri totali: {total_params:,}\n"
    summary += f"  Parametri addestrabili: {trainable_params:,}\n"
    summary += f"  Tipo modello: {model.__class__.__name__}\n"

    return summary
