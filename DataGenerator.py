import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from albumentations import Compose, HorizontalFlip, Rotate, OneOf, RandomScale, Blur, GaussNoise, Resize
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================
# Section 1 : Fonctions utilitaires
# ============================

def load_image(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Charge et redimensionne une image."""
    img = load_img(path, target_size=target_size)
    return img_to_array(img).astype("float32") / 255.0  # Normalisation entre 0 et 1

def load_mask(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Charge et redimensionne un masque."""
    mask = load_img(path, target_size=target_size, color_mode="grayscale")
    return img_to_array(mask).astype("uint8").squeeze()

def one_hot_encode_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Encode le masque en one-hot."""
    one_hot = np.zeros((*mask.shape, num_classes), dtype=np.uint8)
    for class_id in range(num_classes):
        one_hot[..., class_id] = (mask == class_id)
    return one_hot

# ============================
# Section 2 : Définition des augmentations
# ============================

def get_augmentations(image_size: Tuple[int, int]) -> Compose:
    """Définit les transformations Albumentations."""
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=45, p=0.7),
        OneOf([
            RandomScale(scale_limit=0.2, p=0.5),
            Blur(blur_limit=5, p=0.5),
            GaussNoise(std_range=(0.1, 0.5), mean_range=(0.0, 0.0), p=0.5)
        ], p=0.7),
        Resize(*image_size)  # Garantir une taille uniforme
    ])

# ============================
# Section 3 : DataGenerator
# ============================

class DataGenerator(Sequence):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 32,
        num_classes: int = 8,  # Nombre de classes
        shuffle: bool = True,
        augmentation_ratio: float = 1.0
    ):
        """
        Initialise le générateur de données.

        Args:
            image_paths (List[str]): Liste des chemins des images d'entrée.
            mask_paths (List[str]): Liste des chemins des masques.
            image_size (Tuple[int, int]): Taille des images/masks (hauteur, largeur).
            batch_size (int): Taille des lots.
            num_classes (int): Nombre total de classes pour les masques.
            shuffle (bool): Indique si les données doivent être mélangées à chaque epoch.
            augmentation_ratio (float): Ratio d'augmentation [0 à 1]. Définit la proportion des images augmentées.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation_ratio = augmentation_ratio
        self.augmentation = get_augmentations(image_size)
        self.on_epoch_end()

    def __len__(self) -> int:
        """Retourne le nombre de lots par epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère un lot d'images et de masques.

        Args:
            index (int): Index du lot.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (batch_images, batch_masks)
        """
        start = index * self.batch_size
        end = start + self.batch_size
        batch_image_paths = self.image_paths[start:end]
        batch_mask_paths = self.mask_paths[start:end]

        batch_images, batch_masks = [], []

        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            img = load_image(img_path, self.image_size)
            mask = load_mask(mask_path, self.image_size)

            # Appliquer des augmentations selon augmentation_ratio
            if np.random.rand() < self.augmentation_ratio:
                augmented = self.augmentation(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']

            batch_images.append(img)
            batch_masks.append(one_hot_encode_mask(mask, self.num_classes))

        return np.stack(batch_images), np.stack(batch_masks)

    def on_epoch_end(self) -> None:
        """Mélange les données après chaque epoch si shuffle est activé."""
        if self.shuffle:
            data = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(data)
            self.image_paths, self.mask_paths = zip(*data)

    def visualize_batch(self, num_images: int = 5) -> None:
        """
        Visualise les premières images et masques d'un batch.

        Args:
            num_images (int): Nombre d'images/masques à afficher.
        """
        batch_images, batch_masks = self.__getitem__(0)  # Premier lot
        num_images = min(num_images, len(batch_images))
        fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        for i in range(num_images):
            axes[i, 0].imshow(batch_images[i])
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(np.argmax(batch_masks[i], axis=-1), cmap="inferno")  # Décodage pour affichage
            axes[i, 1].set_title("Mask (decoded)")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

# ============================
# Section 4 : Exemple d'utilisation
# ============================

if __name__ == "__main__":
    # Exemple de chemins
    train_input_img_paths = ["path/to/train/image1.jpg", "path/to/train/image2.jpg"]
    train_label_ids_img_paths = ["path/to/train/mask1.png", "path/to/train/mask2.png"]

    # Création du générateur
    train_gen = DataGenerator(
        image_paths=train_input_img_paths,
        mask_paths=train_label_ids_img_paths,
        image_size=(512, 512),
        batch_size=8,
        num_classes=8,
        shuffle=True,
        augmentation_ratio=0.5
    )

    # Visualisation d'un batch
    train_gen.visualize_batch(num_images=3)
