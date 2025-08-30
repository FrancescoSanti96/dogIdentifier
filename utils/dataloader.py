#!/usr/bin/env python3
"""
Utilità per il caricamento dati - Progetto Dog Breed Identifier

Questo modulo fornisce:
1. DogBreedDataset: Dataset personalizzato per classificazione multi-class
2. MyDogDataset: Classificazione binaria per identificazione cane personale
3. Trasformazioni immagini con data augmentation
4. Creazione DataLoader con split train/val/test
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .config_helper import ConfigHelper


class DogBreedDataset(Dataset):
    """
    Dataset personalizzato per classificazione razze canine

    Gestisce caricamento e pre-processing di immagini per classificazione multi-class.
    Supporta caricamento completo del dataset o subset per testing.

    Caratteristiche:
    - Rilevamento automatico cartelle razze e indicizzazione
    - Supporto formati: .jpg, .jpeg, .png
    - Limitazione razze opzionale per test rapidi
    - Statistiche dataset integrate

    Args:
        data_dir (str): Directory contenente cartelle delle razze
        transform (transforms.Compose, optional): Trasformazioni immagini
        max_breeds (int, optional): Numero massimo razze da includere
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_breeds: Optional[int] = None,
        allowed_breeds: Optional[List[str]] = None,
    ):
        """Initialize dataset with automatic breed discovery and loading"""
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []  # Lista percorsi immagini
        self.labels = []  # Lista indici razze (0, 1, 2, ...)
        self.breed_names = []  # Lista nomi razze (nomi cartelle)
        self.max_breeds = max_breeds
        self.allowed_breeds = allowed_breeds

        # Valida che la directory dati esista
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory dati non trovata: {data_dir}")

        # Carica dati e costruisci mapping razze
        self._load_data(max_breeds)
        self._validate_dataset()

    def _load_data(self, max_breeds: Optional[int] = None):
        """
        Load image paths and labels from breed folders

        Questo metodo implementa il discovery automatico delle razze e la creazione
        del mapping breed_name -> label_index. È fondamentale per la scalabilità
        del progetto (5 -> 121 razze) e mantiene consistenza nell'ordinamento.

        Process:
        1. Discovers all breed folders in data_dir (alphabetical order)
        2. Optionally limits to max_breeds for testing
        3. Loads all valid image files from each breed folder
        4. Creates label mapping (breed_name -> index)
        5. Validates dataset consistency and reports statistics

        Args:
            max_breeds (int, optional): Limit number of breeds for quick testing
        """
        # Ottieni tutte le cartelle razze
        available_folders = {
            f.name: f
            for f in self.data_dir.iterdir()
            if f.is_dir() and not f.name.startswith(".")
        }

        # Se è fornito un sottoinsieme con ordine esplicito, applicalo
        if self.allowed_breeds:
            breed_folders = []
            for breed in self.allowed_breeds:
                if breed in available_folders:
                    breed_folders.append(available_folders[breed])
                else:
                    print(f"⚠️  Allowed breed '{breed}' not found in {self.data_dir}")
        else:
            # Default: ordine alfabetico per consistenza
            breed_folders = sorted(
                list(available_folders.values()), key=lambda p: p.name
            )

        if not breed_folders:
            raise ValueError(f"No breed folders found in {self.data_dir}")

        # Limita razze per test se specificato
        if max_breeds and not self.allowed_breeds:
            breed_folders = breed_folders[:max_breeds]
            print(f"🔬 Uso solo le prime {max_breeds} razze per test")

        print(f"📁 Loading {len(breed_folders)} breeds...")

        # Carica immagini da ogni cartella razza
        for breed_idx, breed_folder in enumerate(breed_folders):
            breed_name = breed_folder.name
            self.breed_names.append(breed_name)

            # Ottieni tutti i file immagine supportati
            image_files = self._get_image_files(breed_folder)

            if not image_files:
                print(f"⚠️  No images found in {breed_name} folder")
                continue

            # Aggiungi immagini e etichette
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(breed_idx)

            print(f"   {breed_name}: {len(image_files)} images")

        print(
            f"📊 Total loaded: {len(self.images)} images from {len(self.breed_names)} breeds"
        )

    def _get_image_files(self, folder_path: Path) -> List[Path]:
        """
        Get all valid image files from a folder

        Args:
            folder_path (Path): Path to breed folder

        Returns:
            List[Path]: List of valid image file paths
        """
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []

        for extension in supported_extensions:
            image_files.extend(list(folder_path.glob(extension)))

        return sorted(image_files)  # Ordina per consistenza

    def _validate_dataset(self):
        """
        Validate dataset consistency and report statistics

        Checks:
        - Minimum images per breed
        - Dataset balance
        - Potential issues
        """
        if len(self.images) == 0:
            raise ValueError("No images loaded. Check your data directory structure.")

        # Calcola distribuzione razze
        breed_counts = Counter(self.labels)
        min_images = min(breed_counts.values())
        max_images = max(breed_counts.values())
        avg_images = np.mean(list(breed_counts.values()))

        print(f"\n📈 Dataset Statistics:")
        print(f"   Min images per breed: {min_images}")
        print(f"   Max images per breed: {max_images}")
        print(f"   Avg images per breed: {avg_images:.1f}")

        # Controlla razze severamente sbilanciate
        imbalance_ratio = max_images / min_images if min_images > 0 else float("inf")
        if imbalance_ratio > 5:
            print(
                f"⚠️  Sbilanciamento dataset rilevato! Rapporto: {imbalance_ratio:.1f}:1"
            )
            print("   Considera di bilanciare il dataset per risultati migliori")

        # Mostra razze con più e meno immagini
        sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
        print(
            f"\n🏆 Most images: {self.breed_names[sorted_breeds[0][0]]} ({sorted_breeds[0][1]} images)"
        )
        print(
            f"🔽 Least images: {self.breed_names[sorted_breeds[-1][0]]} ({sorted_breeds[-1][1]} images)"
        )

    def get_breed_names(self) -> List[str]:
        """Get list of breed names in order of label indices"""
        return self.breed_names.copy()

    def get_breed_distribution(self) -> Dict[str, int]:
        """
        Get distribution of images per breed

        Returns:
            Dict[str, int]: Mapping of breed_name -> image_count
        """
        breed_counts = Counter(self.labels)
        return {self.breed_names[idx]: count for idx, count in breed_counts.items()}

    def __len__(self) -> int:
        """Return total number of images in dataset"""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at specified index

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, int]: (transformed_image, breed_label)

        Raises:
            IndexError: If idx is out of range
            IOError: If image cannot be loaded
        """
        if idx >= len(self.images):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.images)}"
            )

        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            # Carica immagine e converti in RGB (rimuove canale alpha se presente)
            image = Image.open(img_path).convert("RGB")

            # Applica trasformazioni se fornite
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")

    def __repr__(self) -> str:
        """String representation of the dataset"""
        return (
            f"DogBreedDataset(num_images={len(self.images)}, "
            f"num_breeds={len(self.breed_names)}, "
            f"data_dir='{self.data_dir}')"
        )


class MyDogDataset(Dataset):
    """Dataset personalizzato per identificazione cane personale (classificazione binaria)"""

    def __init__(self, data_dir: str, transform=None, my_dog_folder="my_dog", other_dogs_folder="other_dogs"):
        """
        Inizializza dataset per cane personale

        Args:
            data_dir: Directory con cartelle del mio cane e altri cani
            transform: Trasformazioni immagini
            my_dog_folder: Nome cartella del mio cane (default: 'my_dog', può essere 'maggie')
            other_dogs_folder: Nome cartella altri cani (default: 'other_dogs', può essere 'other')
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.my_dog_folder = my_dog_folder
        self.other_dogs_folder = other_dogs_folder
        self.images = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """Load image paths and binary labels"""
        my_dog_dir = self.data_dir / self.my_dog_folder
        other_dogs_dir = self.data_dir / self.other_dogs_folder

        # Carica immagini del mio cane (etichetta 1)
        if my_dog_dir.exists():
            my_dog_images = (
                list(my_dog_dir.glob("*.jpg"))
                + list(my_dog_dir.glob("*.jpeg"))
                + list(my_dog_dir.glob("*.png"))
            )

            for img_path in my_dog_images:
                self.images.append(str(img_path))
                self.labels.append(1)

        # Carica immagini altri cani (etichetta 0)
        if other_dogs_dir.exists():
            other_dog_images = (
                list(other_dogs_dir.glob("*.jpg"))
                + list(other_dogs_dir.glob("*.jpeg"))
                + list(other_dogs_dir.glob("*.png"))
            )

            for img_path in other_dog_images:
                self.images.append(str(img_path))
                self.labels.append(0)

        print(
            f"🐕 Personal dog dataset: {sum(self.labels)} {self.my_dog_folder}, {len(self.labels) - sum(self.labels)} {self.other_dogs_folder}"
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and binary label at index"""
        img_path = self.images[idx]
        label = self.labels[idx]

        # Carica e trasforma immagine
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(
    image_size: Tuple[int, int] = (224, 224), augmentation_config: Optional[Dict] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Crea trasformazioni per immagini di training e validazione

    Questa funzione è cruciale per le performance del modello. Crea due pipeline
    separate per massimizzare l'efficacia del training e la consistenza della validazione.

    Design Philosophy:
    1. Training transforms: Data augmentation per generalizzazione e robustezza
    2. Validation transforms: Solo preprocessing (no augmentation) per risultati consistenti
    3. ImageNet normalization: Compatibilità con transfer learning e best practices

    Training Pipeline:
    - RandomResizedCrop: Evita distorsioni aspect ratio + augmentation spaziale
    - Horizontal flip: Simmetria naturale per cani (non cambia la razza)
    - Rotation: Robustezza a orientamenti diversi (limitato per non distorcere)
    - Color jittering: Robustezza a condizioni illuminazione diverse
    - Normalization: ImageNet stats per compatibilità backbone pre-trained

    Validation Pipeline:
    - Resize(256) + CenterCrop(224): Preprocessing deterministico
    - Normalization: Stesse stats del training per consistency

    Args:
        image_size (Tuple[int, int]): Target image size (width, height)
        augmentation_config (Dict, optional): Configuration for data augmentation
            Expected keys:
            - horizontal_flip (bool): Enable random horizontal flipping
            - rotation (int): Maximum rotation degrees
            - brightness_contrast (List[float]): [brightness_factor, contrast_factor]
            - color_jitter (List[float]): [saturation, hue, -, -] (last two unused)
            - random_resized_crop (bool): Use RandomResizedCrop vs Resize+RandomCrop
            - rrc_scale (Tuple[float, float]): Scale range for RandomResizedCrop
            - rrc_ratio (Tuple[float, float]): Aspect ratio range for RandomResizedCrop

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, val_transform)

    Example:
        >>> train_tf, val_tf = get_transforms((224, 224), {
        ...     'horizontal_flip': True,
        ...     'rotation': 15,
        ...     'brightness_contrast': [0.8, 1.2],
        ...     'random_resized_crop': True,
        ...     'rrc_scale': (0.85, 1.0)
        ... })
    """
    # Valori normalizzazione ImageNet - standard per la maggior parte dei modelli pre-addestrati
    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Medie canali RGB
    IMAGENET_STD = [0.229, 0.224, 0.225]  # Deviazioni standard canali RGB

    # Trasformazioni base per validazione (senza augmentation)
    # Usa Resize(256) + CenterCrop per evitare distorsioni aspect ratio
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Trasformazioni training con augmentation opzionale
    if augmentation_config:
        transform_list = []

        # 1. Spatial cropping strategy: RandomResizedCrop or Resize+RandomCrop
        if augmentation_config.get("random_resized_crop", False):
            rrc_scale = augmentation_config.get("rrc_scale", (0.8, 1.0))
            rrc_ratio = augmentation_config.get("rrc_ratio", (0.9, 1.1))
            transform_list.append(
                transforms.RandomResizedCrop(
                    image_size, scale=rrc_scale, ratio=rrc_ratio
                )
            )
        else:
            resize_factor = 1.1  # 10% larger than target
            enlarged_size = (
                int(image_size[0] * resize_factor),
                int(image_size[1] * resize_factor),
            )
            transform_list.append(transforms.Resize(enlarged_size))
            transform_list.append(transforms.RandomCrop(image_size))

        # 2. Random horizontal/vertical flips
        if augmentation_config.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if augmentation_config.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.1))

        # 3. Random rotation
        rotation_degrees = augmentation_config.get("rotation", 0)
        if rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

        # 4. Perspective distortion
        perspective_p = augmentation_config.get("perspective_p", 0.0)
        if perspective_p and perspective_p > 0:
            perspective_scale = augmentation_config.get("perspective_scale", 0.3)
            transform_list.append(
                transforms.RandomPerspective(
                    distortion_scale=perspective_scale, p=perspective_p
                )
            )

        # 5. Color jittering
        brightness_contrast = augmentation_config.get("brightness_contrast", [1.0, 1.0])
        color_jitter = augmentation_config.get("color_jitter", [0.0, 0.0, 0.0, 0.0])
        if brightness_contrast != [1.0, 1.0] or any(
            val > 0 for val in color_jitter[:2]
        ):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=(
                        brightness_contrast[0] if brightness_contrast[0] != 1.0 else 0
                    ),
                    contrast=(
                        brightness_contrast[1] if brightness_contrast[1] != 1.0 else 0
                    ),
                    saturation=color_jitter[0],
                    hue=color_jitter[1],
                )
            )

        # 6. ToTensor + Normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # 7. Random Erasing (applied on tensors)
        erasing_p = augmentation_config.get("erasing_p", 0.0)
        if erasing_p and erasing_p > 0:
            erasing_scale = augmentation_config.get("erasing_scale", (0.02, 0.1))
            erasing_ratio = augmentation_config.get("erasing_ratio", (0.3, 3.3))
            transform_list.append(
                transforms.RandomErasing(
                    p=erasing_p, scale=erasing_scale, ratio=erasing_ratio
                )
            )

        train_transform = transforms.Compose(transform_list)

        print(f"🎨 Data augmentation enabled:")
        print(
            f"   RandomResizedCrop: {augmentation_config.get('random_resized_crop', False)}"
        )
        print(
            f"   Horizontal flip: {augmentation_config.get('horizontal_flip', False)}"
        )
        print(f"   Vertical flip: {augmentation_config.get('vertical_flip', False)}")
        print(f"   Rotation: ±{rotation_degrees}°")
        print(f"   Perspective: p={perspective_p}")
        print(f"   Brightness/Contrast: {brightness_contrast}")
        print(f"   Color jitter (sat/hue): {color_jitter[:2]}")
        print(f"   RandomErasing p: {erasing_p}")

    else:
        # Nessuna augmentation - usa stesse trasformazioni della validazione
        train_transform = val_transform
        print("🎨 Nessuna data augmentation applicata")

    return train_transform, val_transform


def create_dataloaders_from_splits(
    splits_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
    allowed_breeds: Optional[List[str]] = None,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea dataloaders da directory dataset pre-divisi

    Questa funzione carica dati da directory separate train/val/test.
    Questo approccio è migliore degli split casuali perché:
    1. Garantisce valutazione consistente tra esecuzioni
    2. Previene data leakage tra set
    3. Permette analisi statistiche appropriate

    Args:
        splits_dir (str): Directory contenente cartelle train/, val/, test/
        batch_size (int): Dimensione batch per dataloaders
        num_workers (int): Numero di processi worker per caricamento dati
        image_size (Tuple[int, int]): Dimensione immagine target
        augmentation_config (Dict, optional): Configurazione data augmentation
        allowed_breeds (List[str], optional): Lista razze da includere
        use_weighted_sampler (bool): Se usare campionamento pesato

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)

    Struttura directory prevista:
        splits_dir/
        ├── train/
        │   ├── razza1/
        │   └── razza2/
        ├── val/
        │   ├── razza1/
        │   └── razza2/
        └── test/
            ├── razza1/
            └── razza2/

    Esempio:
        >>> train_loader, val_loader, test_loader = create_dataloaders_from_splits('data/splits')
    """
    splits_path = Path(splits_dir)

    # Valida struttura directory
    required_dirs = ["train", "val", "test"]
    for dir_name in required_dirs:
        dir_path = splits_path / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory richiesta non trovata: {dir_path}")

    # Ottieni trasformazioni
    # Assicura che pipeline training usi crop per evitare distorsioni; validazione usa CenterCrop
    # Se augmentation_config non fornito, abilita RandomResizedCrop di default per train
    if augmentation_config is None:
        augmentation_config = {"random_resized_crop": True}
    train_transform, val_transform = get_transforms(image_size, augmentation_config)

    print(f"📁 Caricamento dataset da splits: {splits_dir}")

    # Crea dataset per ogni split
    train_dataset = DogBreedDataset(
        str(splits_path / "train"),
        transform=train_transform,
        allowed_breeds=allowed_breeds,
    )

    val_dataset = DogBreedDataset(
        str(splits_path / "val"), transform=val_transform, allowed_breeds=allowed_breeds
    )

    test_dataset = DogBreedDataset(
        str(splits_path / "test"),
        transform=val_transform,
        allowed_breeds=allowed_breeds,
    )

    # Verifica che tutti i dataset abbiano le stesse razze
    train_breeds = set(train_dataset.get_breed_names())
    val_breeds = set(val_dataset.get_breed_names())
    test_breeds = set(test_dataset.get_breed_names())

    if not (train_breeds == val_breeds == test_breeds):
        print("⚠️  Attenzione: Non tutti gli split contengono le stesse razze!")
        print(f"   Razze train: {len(train_breeds)}")
        print(f"   Razze val: {len(val_breeds)}")
        print(f"   Razze test: {len(test_breeds)}")

        # Trova razze mancanti
        all_breeds = train_breeds | val_breeds | test_breeds
        for split_name, breeds in [
            ("train", train_breeds),
            ("val", val_breeds),
            ("test", test_breeds),
        ]:
            missing = all_breeds - breeds
            if missing:
                print(f"   Mancanti da {split_name}: {sorted(missing)}")

    # Se allowed_breeds fornito, riporta l'ordine imposto
    if allowed_breeds:
        print(f"🔢 Imposizione ordine razze: {allowed_breeds}")

    # Crea DataLoaders - cuore del sistema di caricamento dati
    if use_weighted_sampler:
        # Weighted Sampling per bilanciare classi sbilanciate durante training
        # Obiettivo: ogni classe ha stessa probabilità di essere campionata in ogni epoca
        import numpy as _np

        # Calcola distribuzione classi nel training set
        labels_np = _np.array(train_dataset.labels)
        class_counts = _np.bincount(
            labels_np, minlength=len(train_dataset.breed_names)
        ).astype(_np.float32)
        class_counts[class_counts == 0] = 1.0  # Evita divisione per zero

        # Peso inversamente proporzionale alla frequenza: classi rare pesano di più
        class_weights = (
            class_counts.max() / class_counts
        )  # [max_count/count_i for each class]
        sample_weights = [
            class_weights[label] for label in labels_np
        ]  # Peso per ogni campione

        # WeightedRandomSampler: campiona con probabilità proporzionale ai pesi
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Sampler gestisce l'ordine
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches for consistent training
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"✅ Dataloaders created successfully:")
    print(f"   Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of classes: {len(train_dataset.get_breed_names())}")

    return train_loader, val_loader, test_loader
