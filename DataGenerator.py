import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from albumentations import Compose
from albumentations import (
    HorizontalFlip, Rotate, RandomScale, Blur, GaussNoise, 
    GridDistortion, ElasticTransform, CoarseDropout, 
    RandomBrightnessContrast, OneOf, Resize
)
from albumentations.augmentations.transforms import RandomFog, RandomRain
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================
# Section 1 : Fonctions utilitaires
# ============================

def load_image(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    img = load_img(path, target_size=target_size)
    return img_to_array(img).astype("float32") / 255.0  # Normalisation [0,1]

def load_mask(path: str, target_size: Tuple[int, int]) -> np.ndarray:
    mask = load_img(path, target_size=target_size, color_mode="grayscale")
    return img_to_array(mask).astype("uint8").squeeze()

def one_hot_encode_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((*mask.shape, num_classes), dtype=np.uint8)
    for class_id in range(num_classes):
        one_hot[..., class_id] = (mask == class_id)
    return one_hot

# ============================
# Section 2 : Définition des augmentations
# ============================

def get_augmentations(image_size: Tuple[int, int]) -> Compose:
    return Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.7),
        OneOf([
            RandomScale(scale_limit=0.2, p=0.5),
            Blur(blur_limit=5, p=0.5),
            GaussNoise(std_range=(0.2, 0.44), mean_range=(0.0, 0.0), per_channel=True,                       noise_scale_factor=1, p=0.5)   
        ], p=0.7),
        
        # Améliorations pour objets fins
        ElasticTransform(alpha=1, sigma=50, p=0.3),
        GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2),
        hole_width_range=(0.1, 0.2), fill=0, fill_mask=None, p=0.5),  

        # Effets météo
        OneOf([
            RandomFog(alpha_coef=0.08, fog_coef_range=(0.3, 1), p=0.3),
            RandomRain(drop_width=1, blur_value=3, p=0.3)  
        ], p=0.3),

        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        
        Resize(*image_size)
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
        num_classes: int = 8,
        shuffle: bool = True,
        augmentation_ratio: float = 1.0,
        use_sample_weights: bool = True,
        dynamic_class_weights: bool = True,  # Option pour activer la balance dynamique
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation_ratio = augmentation_ratio
        self.use_sample_weights = use_sample_weights
        self.dynamic_class_weights = dynamic_class_weights  # Option pour la balance dynamique
        self.augmentation = get_augmentations(image_size) if augmentation_ratio > 0 else None
        self.epoch = 0  # ✅ Option démarre à -1 pour éviter un premier `on_epoch_end()`
#        self.on_epoch_end()  # ✅ Permet de bien démarrer avec `epoch = 0`

    def __len__(self) -> int:
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int):
        #print(f"DEBUG - Appel de __getitem__, epoch actuel: {self.epoch}")
        batch_images, batch_masks = self._generate_batch(index)
        
        if self.use_sample_weights:
            class_weights = self._compute_class_weights(batch_masks)
            if self.dynamic_class_weights:
                class_weights = self._adjust_class_weights(class_weights)
            sample_weights = self._compute_sample_weights(batch_masks, class_weights)
            return batch_images, batch_masks, sample_weights
        else:
            return batch_images, batch_masks

    def _generate_batch(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        end = start + self.batch_size
        batch_image_paths = self.image_paths[start:end]
        batch_mask_paths = self.mask_paths[start:end]
    
        # ✅ Vérification que des fichiers sont bien sélectionnés
        if len(batch_image_paths) == 0 or len(batch_mask_paths) == 0:
            print(f"⚠️ Aucun fichier trouvé pour index {index} ! Batch vide ignoré.")
            return np.array([]), np.array([])  
    
        batch_images, batch_masks = [], []
    
        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            img = load_image(img_path, self.image_size)
            mask = load_mask(mask_path, self.image_size)
    
            if self.augmentation and np.random.rand() < self.augmentation_ratio:
                augmented = self.augmentation(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
    
            batch_images.append(img)
            batch_masks.append(one_hot_encode_mask(mask, self.num_classes))
    
        # ✅ Vérification APRÈS la boucle
        if len(batch_images) == 0 or len(batch_masks) == 0:
            print(f"⚠️ Batch vide détecté après chargement pour index {index}, aucun traitement effectué.")
            return np.array([]), np.array([])  
    
        # print(f"DEBUG - Index demandé : {index}")
        # print(f"DEBUG - Nombre d'images dans le batch : {len(batch_images)}")
        # print(f"DEBUG - Nombre de masques dans le batch : {len(batch_masks)}")
    
        return np.stack(batch_images), np.stack(batch_masks)


    def _compute_class_weights(self, batch_masks: np.ndarray) -> np.ndarray:
        """
        Calcul des poids par classe à la volée
        Si c'est le premier epoch, on retourne un poids égal pour toutes les classes.
        Sinon, on retourne des poids dynamiques basés sur la fréquence des classes dans le batch.
        """
#        if self.epoch == 0:
            # Premier epoch, tous les poids égaux (pas d'influence sur la perte)
#            return np.ones(self.num_classes)
        
        # Si ce n'est pas le premier epoch, calcul des poids par classe
        pixel_counts = np.sum(batch_masks, axis=(0, 1, 2))
        # Éviter les divisions par 0 (si une classe est absente)
        pixel_counts = np.maximum(pixel_counts, 1)
        
        # Calcul des poids en inversant les fréquences (plus rare = poids plus grand)
        class_weights = np.sum(pixel_counts) / pixel_counts  # Calcul des poids des classes
        # return class_weights / np.sum(class_weights)  # Normalisation classique par la somme

        # Mise à l'échelle logarithmique (en ajoutant un epsilon pour éviter log(0))
        epsilon = 1e-2  # Ajout d'un petit epsilon pour éviter log(0)
        log_class_weights = np.log(class_weights + epsilon)  # Logarithme des poids

        # Mise à l'échelle pour avoir des poids compris entre 1 et xx
        min_weight = 1
        max_weight = 5
    
        # Normalisation linéaire des poids log-transformés
        scale_factor = (max_weight - min_weight) / (np.max(log_class_weights) - np.min(log_class_weights))
        scaled_class_weights = (log_class_weights - np.min(log_class_weights)) * scale_factor + min_weight
        
        # Conversion en entiers (arrondi)
        scaled_class_weights = np.round(scaled_class_weights).astype(int)
    
        return scaled_class_weights


    def _adjust_class_weights(self, class_weights: np.ndarray) -> np.ndarray:
        """
        Ajuste les poids des classes au fur et à mesure des époques.
        Au début, les poids sont tous égaux, et ils augmentent progressivement pour compenser
        les classes rares.
        """
#        if self.epoch == 0:  # ✅ Premier epoch, pas de modification
#            return np.ones_like(class_weights)
    
        # Facteur d'augmentation basé sur l'époque
        epoch_factor = 1 + 0.05 * self.epoch  # Facteur d'augmentation progressif
        adjusted_weights = class_weights * epoch_factor
        return adjusted_weights

    def _compute_sample_weights(self, batch_masks: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
        """Calcule les poids de classe pour chaque pixel."""
        #print(f"DEBUG - Epoch: {self.epoch}")  # ✅ Ajout de debug
        
        # Possibilité d'utiliser une répartition de classes statique
       #  class_weights = [0.04731172, 0.0125408 , 0.02223516, 0.2754033 , 0.03222498,
       # 0.13717382, 0.40695924, 0.06615099]
        
#        if self.epoch == 0:
            #print("DEBUG - Premier epoch, sample_weights doit être 1 partout")  # ✅ Vérification
#            return np.ones(batch_masks.shape[:-1])  # ✅ Devrait être (batch_size, H, W) 
            
        weights = np.dot(batch_masks, class_weights)
        return weights

    def on_epoch_end(self) -> None:
        """Incrémente l'époch et réinitialise les poids si nécessaire."""
#        if self.epoch == -1:
#            self.epoch = 0  # ✅ On met à 0 au premier appel
#        else:
        self.epoch += 1 # Incrémenter l'époch
        
        if self.shuffle:
            data = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(data)
            self.image_paths, self.mask_paths = zip(*data)

    def visualize_batch(self, num_images: int = 5) -> None:
        batch_images, batch_masks = self.__getitem__(0)[:2]
        num_images = min(num_images, len(batch_images))
        fig, axes = plt.subplots(num_images, 2, figsize=(6, num_images * 3))

        for i in range(num_images):
            axes[i, 0].imshow(batch_images[i])
            axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(np.argmax(batch_masks[i], axis=-1), cmap="inferno")
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

    val_input_img_paths = ["path/to/val/image1.jpg", "path/to/val/image2.jpg"]
    val_label_ids_img_paths = ["path/to/val/mask1.png", "path/to/val/mask2.png"]

    # Création des générateurs
    train_gen = DataGenerator(
        image_paths=train_input_img_paths,
        mask_paths=train_label_ids_img_paths,
        batch_size=8,
        num_classes=8,
        shuffle=True,
        augmentation_ratio=0.5,     # ratio pour limiter les classes rare (si 1 pas d'effet)
        use_sample_weights=True,    # ✅ Utilise sample_weights pour l'entraînement
        dynamic_class_weights=True  # Active la balance dynamique des poids pendant l'entrainement
    )
    

    # Visualisation d'un batch
    train_gen.visualize_batch(num_images=3)
