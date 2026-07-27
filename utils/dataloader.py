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
        Carica percorsi immagini e label dalle cartelle delle razze.

        Questo metodo esegue il discovery delle cartelle razze e costruisce
        il mapping `breed_name -> label_index`, mantenendo un ordinamento
        consistente. È pensato per scalare senza modifiche da 5 a 121 razze.

        Passi:
        1. Individua tutte le cartelle razze in `data_dir` (ordine alfabetico)
        2. Opzionalmente limita a `max_breeds` per test rapidi
        3. Carica i file immagine validi per ogni razza
        4. Crea le etichette numeriche coerenti con l'ordine delle razze
        5. Stampa statistiche sintetiche del dataset

        Args:
            max_breeds (int, optional): Limite massimo di razze (debug/test)
        """
        # Discovery automatico cartelle razze: escludi file nascosti e temp
        available_folders = {
            f.name: f
            for f in self.data_dir.iterdir()
            if f.is_dir() and not f.name.startswith(".")  # Ignora .DS_Store, .tmp, etc.
        }

        # Gestione subset razze: ordine esplicito vs alfabetico
        if self.allowed_breeds:
            # Usa ordine specificato (per subset target come top_5, top_10)
            breed_folders = []
            for breed in self.allowed_breeds:
                if breed in available_folders:
                    breed_folders.append(available_folders[breed])
                else:
                    print(f"⚠️  Allowed breed '{breed}' not found in {self.data_dir}")
        else:
            # Default: ordine alfabetico per consistenza tra runs
            breed_folders = sorted(
                list(available_folders.values()), key=lambda p: p.name
            )

        if not breed_folders:
            raise ValueError(f"No breed folders found in {self.data_dir}")

        # Limita razze per test rapidi se specificato (sovrascritto da allowed_breeds)
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
        Ritorna tutti i file immagine validi presenti in una cartella.

        Args:
            folder_path (Path): Percorso alla cartella della razza

        Returns:
            List[Path]: Percorsi dei file immagine trovati
        """
        supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_files = []

        for extension in supported_extensions:
            image_files.extend(list(folder_path.glob(extension)))

        return sorted(image_files)  # Ordina per consistenza

    def _validate_dataset(self):
        """
        Valida la consistenza del dataset e stampa statistiche chiave.

        Controlli effettuati:
        - Numero minimo di immagini per razza
        - Sbilanciamento tra razze (rapporto max/min)
        - Razze con più e meno immagini
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
        """Ritorna la lista dei nomi delle razze nell'ordine delle label."""
        return self.breed_names.copy()

    def get_breed_distribution(self) -> Dict[str, int]:
        """
        Ritorna la distribuzione del numero immagini per razza.

        Returns:
            Dict[str, int]: Mapping `breed_name -> image_count`.
        """
        breed_counts = Counter(self.labels)
        return {self.breed_names[idx]: count for idx, count in breed_counts.items()}

    def __len__(self) -> int:
        """Numero totale di immagini nel dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Restituisce immagine e label all'indice specificato.

        Args:
            idx (int): Indice del campione da recuperare

        Returns:
            Tuple[torch.Tensor, int]: (immagine_trasformata, label_razza)

        Raises:
            IndexError: se `idx` è fuori range
            IOError: se l'immagine non può essere caricata
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
        """Rappresentazione testuale del dataset."""
        return (
            f"DogBreedDataset(num_images={len(self.images)}, "
            f"num_breeds={len(self.breed_names)}, "
            f"data_dir='{self.data_dir}')"
        )


class MyDogDataset(Dataset):
    """Dataset personalizzato per identificazione cane personale (classificazione binaria)."""

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
        """Carica percorsi immagini e label binarie."""
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
        """Ritorna immagine e label binaria all'indice specificato."""
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
    Crea pipeline di trasformazioni per training e validazione.

    La pipeline di training include data augmentation opzionale per migliorare
    generalizzazione e robustezza; quella di validazione esegue solo
    preprocessing deterministico per risultati comparabili.

    Args:
        image_size (Tuple[int, int]): Dimensione target immagine (width, height)
        augmentation_config (Dict, optional): Configurazione augmentation. Chiavi attese:
            - horizontal_flip (bool)
            - rotation (int)
            - brightness_contrast (List[float])
            - color_jitter (List[float])
            - random_resized_crop (bool)
            - rrc_scale (Tuple[float, float])
            - rrc_ratio (Tuple[float, float])

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, val_transform)
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
