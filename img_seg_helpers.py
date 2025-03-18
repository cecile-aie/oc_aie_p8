import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import register_keras_serializable
from segmentation_models.losses import DiceLoss
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
from tensorflow.keras.callbacks import Callback
from matplotlib.colors import ListedColormap
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
import cv2  # Utilisé pour charger les images

##### METRIQUE #################################################

@register_keras_serializable(package="Custom", name="IoUMean")
class IoUMean(Metric):
    def __init__(self, name="iou_mean", smooth=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total_iou / (self.count + self.smooth)

    def reset_state(self):
        self.total_iou.assign(0)
        self.count.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config

##### PERTE #################################################

@register_keras_serializable(package="Custom", name="DiceLoss")
class CustomDiceLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, beta=1.0, reduction="sum_over_batch_size", name="dice_loss"):
        super().__init__(reduction=reduction, name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32) if class_weights is not None else None
        self.beta = beta

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denominator = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])

        dice = (2. * intersection + 1e-7) / (denominator + 1e-7)
        dice_loss = 1 - dice

        # Appliquer les poids de classe
        if self.class_weights is not None:
            dice_loss *= self.class_weights

        # Si `sample_weight` est fourni, l’appliquer à la perte
        if sample_weight is not None:
            dice_loss *= sample_weight

        return tf.reduce_mean(dice_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist() if self.class_weights is not None else None,
            "beta": self.beta
        })
        return config

# Instancier une perte par défaut pour éviter l'import manuel
dice_loss = CustomDiceLoss()

# ---------------------------------------------------------------------------------------------------#

@register_keras_serializable(package="Custom", name="TverskyLoss")
class TverskyLoss(Loss):
    def __init__(self, alpha=0.7, beta=0.3, reduction="sum_over_batch_size", name="tversky_loss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # Éviter division par zéro

        # Vérifier si y_true est one-hot, sinon le convertir
        if len(y_true.shape) == 3:  # (batch_size, height, width)
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1]) 

        # Calculer TP, FP et FN pixel par pixel
        TP = y_true * y_pred
        FP = (1 - y_true) * y_pred
        FN = y_true * (1 - y_pred)

        # Réduction sur les axes de l'image et des classes, mais pas sur le batch
        TP = tf.reduce_sum(TP, axis=-1)  # Garder (batch_size, height, width)
        FP = tf.reduce_sum(FP, axis=-1)
        FN = tf.reduce_sum(FN, axis=-1)

        # Calculer l'indice de Tversky par pixel
        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + 1e-7)

        # Retourner une perte compatible avec sample_weights (batch, height, width)
        return 1 - tversky_index

    def get_config(self):
        """Permet la sauvegarde et rechargement automatique avec load_model()."""
        config = super().get_config()
        config.update({"alpha": self.alpha, "beta": self.beta})
        return config

# ---------------------------------------------------------------------------------------------------#

@register_keras_serializable(package="Custom", name="FocalLoss")
class FocalLoss(Loss):
    def __init__(self, gamma=2., alpha=0.25, name="FocalLoss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Calcul de l'entropie croisée
        cross_entropy = -y_true * K.log(y_pred + K.epsilon())
        
        # Calcul du poids focal
        weight = self.alpha * K.pow(1 - y_pred, self.gamma)
        
        # Application du poids sur l'entropie croisée
        focal_loss = K.sum(weight * cross_entropy, axis=-1)
        return focal_loss

    def get_config(self):
        """ Permet d'enregistrer correctement les hyperparamètres gamma et alpha """
        return {"gamma": self.gamma, "alpha": self.alpha}


##### ENTRAINEMENT (visualisation, callbacks) #################################################

import numpy as np
import matplotlib.pyplot as plt

def display_target_image(val_gen, target_image_index):
    """
    Affiche une image spécifique et son masque associé à partir d'un générateur de validation.
    
    :param val_gen: Générateur fournissant des lots d'images et de masques
    :param target_image_index: Index global de l'image cible
    :return: L'image et le masque sélectionnés
    """
    image_count = 0
    fixed_image = None
    fixed_mask = None
    
    for sample_image, sample_mask in val_gen:
        if image_count + len(sample_image) > target_image_index:
            index_in_lot = target_image_index - image_count
            fixed_image = sample_image[index_in_lot]
            fixed_mask = sample_mask[index_in_lot]
            break
        image_count += len(sample_image)
    
    if fixed_image is None or fixed_mask is None:
        print("Index hors limites du générateur.")
        return None, None
    
    # Convertir le masque en format 2D
    fixed_mask_visu = np.argmax(fixed_mask, axis=-1)
    
    # Affichage
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    ax[0].imshow(fixed_image)
    ax[0].set_title("Image Fixe")
    ax[0].axis('off')
    
    ax[1].imshow(fixed_mask_visu, cmap='jet')
    ax[1].set_title("Masque Fixe")
    ax[1].axis('off')
    
    plt.show()
    
    return fixed_image, fixed_mask

# ---------------------------------------------------------------------------------------------------#


class MaskVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, sample_image, sample_mask, save_path="./training_masks"):
        self.sample_image = sample_image
        self.sample_mask = sample_mask
        self.save_path = save_path

        # Vérifier si le répertoire existe, sinon le créer
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"Répertoire '{self.save_path}' créé.")
        else:
            print(f"Répertoire '{self.save_path}' déjà existant.")

    def on_epoch_end(self, epoch, logs=None):
        # Prédiction du modèle
        prediction = self.model.predict(np.expand_dims(self.sample_image, axis=0))[0]

        # Convertir en NumPy si nécessaire
        if isinstance(self.sample_mask, tf.Tensor):
            self.sample_mask = self.sample_mask.numpy()
        
        if isinstance(prediction, tf.Tensor):
            prediction = prediction.numpy()
        
        # Applatissement des masques et des prédictions
        mask_argmax = self.sample_mask.argmax(axis=-1)  # Masque réel aplati
        pred_argmax = prediction.argmax(axis=-1)        # Masque prédit aplati

        # Affichage de l'image, du masque réel et du masque prédit
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Image originale
        axes[0].imshow(self.sample_image, cmap='gray')
        axes[0].set_title("Image")

        # Masque réel
        axes[1].imshow(mask_argmax, cmap='jet')
        axes[1].set_title("Masque Réel")

        # Masque prédit
        axes[2].imshow(pred_argmax, cmap='jet')
        axes[2].set_title("Masque Prédit")

        plt.tight_layout()

        # Sauvegarde de l'image dans le répertoire
        plt.savefig(f"{self.save_path}/mask_epoch_{epoch}.png")

        # Affichage de la figure dans le notebook
        plt.show()

        # Fermeture de la figure
        plt.close(fig)

# ---------------------------------------------------------------------------------------------------#


class EmbeddingLogger(Callback):
    def __init__(self, log_dir, generator, layer_name, num_images=100, batch_size=16, 
                 class_names=None, class_colors=None):
        super().__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.generator = generator
        self.layer_name = layer_name
        self.num_images = num_images
        self.batch_size = batch_size
        self.class_names = class_names or ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']
        self.class_colors = class_colors or {
            0: [0, 0, 0],      # 'void'
            1: [255, 0, 0],    # 'flat'
            2: [0, 255, 0],    # 'construction'
            3: [0, 0, 255],    # 'object'
            4: [255, 255, 0],  # 'nature'
            5: [0, 255, 255],  # 'sky'
            6: [255, 0, 255],  # 'human'
            7: [128, 128, 128] # 'vehicle'
        }
        self.image_dir = os.path.join(log_dir, "projector_images")
        os.makedirs(self.image_dir, exist_ok=True)

    @tf.function(reduce_retracing=True)  # Désactive le re-tracing inutile
    def get_embeddings(self, model, images):
        """Exécute predict() en évitant le re-tracing de TensorFlow."""
        return model(images, training=False)  # training=False pour éviter le mode dropout/batch norm
        
    def get_dominant_class(self, mask):
        class_counts = np.sum(mask, axis=(0, 1))  # Compter les pixels par classe
        dominant_class = np.argmax(class_counts)  # Prendre la classe majoritaire
        return self.class_names[dominant_class]
    
    def get_minority_class(self, mask):
        class_counts = np.sum(mask, axis=(0, 1))
        nonzero_classes = np.where(class_counts > 0)[0]
        if len(nonzero_classes) > 0:
            minority_class = nonzero_classes[np.argmin(class_counts[nonzero_classes])]
            return self.class_names[minority_class]
        return 'void'
    
    def colorize_mask(self, mask):
        colored_masks = []
        for i in range(mask.shape[0]):
            single_mask = np.argmax(mask[i], axis=-1)  # Index des classes
            colored_mask = np.zeros((single_mask.shape[0], single_mask.shape[1], 3), dtype=np.uint8)
            for class_id, color in self.class_colors.items():
                colored_mask[single_mask == class_id] = color  
            colored_masks.append(colored_mask)
        return np.array(colored_masks)
    
    def create_sprite_image(self, image_paths):
        # Charger la première image pour déterminer la taille correcte
        sample_image = Image.open(image_paths[0])
        image_size = sample_image.size  # Détecte la vraie taille
        
        images = [Image.open(img_path).resize(image_size) for img_path in image_paths]
        
        grid_size = int(np.ceil(np.sqrt(len(images))))  # Taille de la grille (carré le plus proche)
        sprite_image = Image.new('RGB', (grid_size * image_size[0], grid_size * image_size[1]))
        
        for i, img in enumerate(images):
            row = i // grid_size
            col = i % grid_size
            sprite_image.paste(img, (col * image_size[0], row * image_size[1]))
        
        sprite_path = os.path.join(self.log_dir, 'sprite.png')
        sprite_image.save(sprite_path)
        return sprite_path, image_size
    
    def on_epoch_end(self, epoch, logs=None):
        embeddings_all, image_paths, metadata = [], [], []
        collected_images = 0   
        data_iter = iter(self.generator)  # Créer un itérateur unique
        
        while collected_images < self.num_images:
            try:
                batch = next(data_iter)  # Utiliser l'itérateur existant
            except StopIteration:
                print("⚠️ Générateur épuisé, on le réinitialise.")
                data_iter = iter(self.generator)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    print("⚠️ Plus d'images disponibles, arrêt de la collecte.")
                    break
            
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                print("⚠️ Batch invalide détecté, on l'ignore.")
                continue
            
            images, true_masks = batch[:2]
            
            # Vérifier si le batch est vide avant de l'utiliser
            if images.shape[0] == 0 or true_masks.shape[0] == 0:
                print(f"⚠️ Batch vide détecté, index {collected_images}, on l'ignore.")
                continue
            
            current_batch_size = min(images.shape[0], self.num_images - collected_images)
            images, true_masks = images[:current_batch_size], true_masks[:current_batch_size]
    
            colored_masks = self.colorize_mask(true_masks)
    
            intermediate_layer_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(self.layer_name).output
            )
            
            embeddings = self.get_embeddings(intermediate_layer_model, images)
            
            if len(embeddings.shape) > 2:
                embeddings = tf.reshape(embeddings, [tf.shape(embeddings)[0], -1])

            for i in range(current_batch_size):
                img_pil = Image.fromarray((images[i] * 255).astype(np.uint8))
                mask_pil = Image.fromarray(colored_masks[i])
    
                img_path = os.path.join(self.image_dir, f"img_{collected_images}.png")
                mask_path = os.path.join(self.image_dir, f"mask_{collected_images}.png")
    
                img_pil.save(img_path)
                mask_pil.save(mask_path)
                image_paths.append(img_path)
    
                dominant_class = self.get_dominant_class(true_masks[i])
                minority_class = self.get_minority_class(true_masks[i])
                metadata.append(f"{collected_images}\t{dominant_class}\t{minority_class}")
    
                collected_images += 1
                if collected_images >= self.num_images:
                    break
    
            embeddings_all.append(embeddings)
    
        embeddings_all = np.concatenate(embeddings_all, axis=0)
    
        with self.file_writer.as_default():
            # Extraire les biais de la couche spécifiée
#            layer = self.model.get_layer(self.layer_name)
#            if hasattr(layer, "bias") and layer.bias is not None:
#                biases = layer.bias.numpy()
#                # Normalisation entre 0 et 255 pour améliorer le contraste
#                biases_rescaled = 255 * (biases - np.min(biases)) / (np.max(biases) - np.min(biases) + 1e-8)
#                biases_rescaled = biases_rescaled.reshape(1, -1, 1)  # 1 ligne, N valeurs, 1 canal
#                biases_rescaled = biases_rescaled.astype(np.uint8)
#                tf.summary.image("bias/image", np.expand_dims(biases_rescaled, axis=0), step=epoch)
#                print(f"✔️ Image des biais enregistrée pour la couche {self.layer_name}.")
        
            tf.summary.scalar("Embeddings Mean", np.mean(embeddings_all), step=epoch)
            tf.summary.histogram("Embeddings", embeddings_all, step=epoch)
            tf.summary.image("Images d'Exemple", np.array([np.array(Image.open(p)) for p in image_paths[:10]]) / 255.0, step=epoch)
            
        checkpoint_path = os.path.join(self.log_dir, f"embeddings_epoch_{epoch}.ckpt")
        checkpoint = tf.train.Checkpoint(embeddings=tf.Variable(embeddings_all))
        checkpoint.save(file_prefix=checkpoint_path)
        with open(os.path.join(self.log_dir, "checkpoint"), "w") as f:
            f.write(f'model_checkpoint_path: "embeddings_epoch_{epoch}.ckpt-1"\n')
    
        metadata_path = os.path.join(self.log_dir, "metadata.tsv")
        with open(metadata_path, "w") as f:
            f.write("index\tdominant_class\tminority_class\n")
            f.writelines("\n".join(metadata) + "\n")
    
        print(f"✔️ metadata.tsv mis à jour pour l'epoch {epoch}")
    
        # Générer l'image sprite
        sprite_path, image_size = self.create_sprite_image(image_paths)
        self.generate_projector_config(image_size)
        print(f"✔️ Sprite image créée et sauvegardée à {sprite_path}")
    
        print(f"✔️ Embeddings et images sauvegardés pour l'epoch {epoch}.")
    
    def generate_projector_config(self, image_size):
        config_path = os.path.join(self.log_dir, "projector_config.pbtxt")
        with open(config_path, "w") as f:
            f.write(f"""
embeddings {{
    tensor_name: "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    metadata_path: "metadata.tsv"
    sprite {{
        image_path: "sprite.png"
        single_image_dim: {image_size[0]}
        single_image_dim: {image_size[1]}
    }}
}}
            """)
        print(f"✔️ Config Projector générée avec taille {image_size[0]}x{image_size[1]} : {config_path}")

# ---------------------------------------------------------------------------------------------------#

def create_embedding_logger(log_dir, generator, model, layer_name, num_images=100, batch_size=16, class_colors=None):
    """
    Vérifie si layer_name existe dans model. Si non, sélectionne une couche de convolution au 1/3 de la profondeur.
    
    Args:
        log_dir (str): Répertoire de logs.
        generator: Générateur de validation.
        model (tf.keras.Model): Modèle de segmentation.
        layer_name (str): Nom de la couche à utiliser pour les embeddings.
        num_images (int): Nombre d'images à analyser.
        batch_size (int): Taille du batch.
        class_colors (dict): Dictionnaire associant les classes aux couleurs.

    Returns:
        EmbeddingLogger: Instance de la classe `EmbeddingLogger` avec une couche validée.
    """
    # Liste des noms de couches dans le modèle
    layer_names = [layer.name for layer in model.layers]
    
    # Vérifier si layer_name existe dans le modèle
    if layer_name not in layer_names:
        print(f"⚠️ La couche '{layer_name}' n'existe pas dans le modèle. Sélection d'une couche alternative...")

        # Filtrer les couches de convolution uniquement
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        
        if not conv_layers:
            raise ValueError("❌ Aucun couche de convolution trouvée dans le modèle.")

        # Sélectionner une couche située au 1/3 de la liste des convolutions
        index = len(conv_layers) // 3
        layer_name = conv_layers[index]

        print(f"✅ Nouvelle couche sélectionnée : '{layer_name}'")

    # Créer et retourner l'instance de EmbeddingLogger
    embedding_logger = EmbeddingLogger(
        log_dir=log_dir,
        generator=generator,
        layer_name=layer_name,
        num_images=num_images,
        batch_size=batch_size,
        class_colors=class_colors
    )

    return embedding_logger

# ---------------------------------------------------------------------------------------------------#

class ImagePredictionLogger(Callback):
    def __init__(self, log_dir, generator, num_classes, num_images=3):
        super().__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.generator = generator
        self.num_classes = num_classes
        self.num_images = num_images  # Nombre d'exemples à enregistrer

        # Définition d'une palette de couleurs pour les classes
        self.colors = np.array([
            [0, 0, 0],        # Classe 0: Noir (void)
            [128, 64, 128],   # Classe 1: Mauve (flat)
            [70, 70, 70],     # Classe 2: Gris foncé (construction)
            [102, 102, 156],  # Classe 3: Bleu-gris (object)
            [107, 142, 35],   # Classe 4: Vert (nature)
            [70, 130, 180],   # Classe 5: Bleu (sky)
            [220, 20, 60],    # Classe 6: Rouge (human)
            [0, 0, 142]       # Classe 7: Bleu foncé (vehicle)
        ], dtype=np.uint8)

        # Création de la colormap personnalisée pour affichage avec Matplotlib
        self.cmap = ListedColormap(self.colors / 255.0)

    def decode_mask(self, mask):
        """Convertit un masque one-hot en une image couleur."""
        mask = np.argmax(mask, axis=-1)  # Convertit one-hot en index de classe
        return self.colors[mask]  # Associe chaque index à une couleur

    def on_epoch_end(self, epoch, logs=None):
        # Obtenir un batch d'images et de masques
        images, masks = next(iter(self.generator))  # Récupère un batch
        preds = self.model.predict(images)  # Prédictions du modèle

        with self.file_writer.as_default():
            for i in range(min(self.num_images, images.shape[0])):
                # Décoder les masques
                true_mask = self.decode_mask(masks[i])
                pred_mask = self.decode_mask(preds[i])

                # Créer une figure avec matplotlib
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(images[i])
                axes[0].set_title("Image")
                axes[1].imshow(true_mask)
                axes[1].set_title("Masque Vérité")
                axes[2].imshow(pred_mask)
                axes[2].set_title("Masque Prédit")

                for ax in axes:
                    ax.axis("off")

                # Convertir la figure en image et l'enregistrer
                plt.tight_layout()
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA
                image = image[:, :, :3]  # Garder seulement les 3 canaux RGB
                plt.close(fig)

                # Ajouter l'image à TensorBoard
                tf.summary.image(f"Exemple {i}", np.expand_dims(image, axis=0), step=epoch)

# ---------------------------------------------------------------------------------------------------#

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, relative_threshold=0.01, patience=5):  
        super(CustomEarlyStopping, self).__init__()
        self.relative_threshold = relative_threshold
        self.patience = patience
        self.best_loss = float('inf')  # Meilleure perte rencontrée
        self.wait = 0  

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        current_loss = logs.get("val_loss")

        if current_loss is None:
            return

        # Calcul du changement relatif par rapport à la meilleure perte observée
        delta_loss = self.best_loss - current_loss
        relative_change = delta_loss / self.best_loss if self.best_loss > 0 else 0  

        if delta_loss >= 0:
            if relative_change < self.relative_threshold:
                self.wait += 1
                print(f"\n⚠️ Avertissement : amélioration relative ({relative_change:.2%}) inférieure au seuil ({self.relative_threshold:.2%}) "
                      f"({self.wait}/{self.patience} epochs)")
            else:
                self.wait = 0  # Reset patience si on améliore assez
                self.best_loss = current_loss  # Mise à jour de la meilleure perte

        if self.wait >= self.patience:
            print("\n🛑 Arrêt de l'entraînement : patience atteinte sans amélioration significative")
            self.model.stop_training = True

# ---------------------------------------------------------------------------------------------------#

def plot_loss_iou(history):
    # Récupération des données d'historique
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    iou = history.history['iou_mean']
    val_iou = history.history['val_iou_mean']
    _epochs = list(range(1, len(loss) + 1))

    # Tracé des courbes de perte
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(_epochs, loss, 'y', label='Perte Training')
    plt.plot(_epochs, val_loss, 'r', label='Perte Validation')
    plt.title('Perte Training et Validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    
    # Tracé des courbes de l'IoU
    plt.subplot(1, 2, 2)
    plt.plot(_epochs, iou, 'y', label='IoU Training')
    plt.plot(_epochs, val_iou, 'r', label='IoU Validation')
    plt.title('IoU moyen Training et Validation')
    plt.xlabel('Époques')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------#


def get_feature_extractor(model):
    """
    Crée un modèle tronqué qui extrait les sorties des couches les plus intéressantes :
    - Convolutions (`Conv2D`, `SeparableConv2D`)
    - Normalisation (`BatchNormalization`)
    - Activation (`Activation`)
    - Pooling (`MaxPooling2D`, `GlobalAveragePooling2D`)
    - Fusion (`Add`, `Concatenate`, `Multiply`)
    """
    # Sélection des couches utiles pour l'affichage
    layers_of_interest = (tf.keras.layers.Conv2D, 
                          tf.keras.layers.SeparableConv2D, 
                          tf.keras.layers.BatchNormalization, 
                          tf.keras.layers.Activation, 
                          tf.keras.layers.MaxPooling2D, 
                          tf.keras.layers.GlobalAveragePooling2D, 
                          tf.keras.layers.Add, 
                          tf.keras.layers.Concatenate, 
                          tf.keras.layers.Multiply)

    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers_of_interest)]

    # Créer le modèle tronqué
    return tf.keras.Model(inputs=model.input, outputs=layer_outputs)


def plot_feature_maps(model, image, mask, num_rows=3, num_cols=8):
    """
    Affiche les cartes de caractéristiques extraites par le modèle pour une image et un masque donnés.
    Fonctionne pour tout modèle basé sur TensorFlow/Keras.
    """
    # Si le masque est en format one-hot, convertir en indices de classes
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask = np.argmax(mask, axis=-1)

    # Étape 1 : Créer un extracteur de caractéristiques pour le modèle
    feature_extractor = get_feature_extractor(model)

    # Étape 2 : Passer l'image à travers le modèle tronqué
    feature_maps = feature_extractor.predict(np.expand_dims(image, axis=0))

    # Vérification et conversion en liste si besoin
    if not isinstance(feature_maps, list):
        feature_maps = [feature_maps]  # Conversion si un seul élément

    # Étape 3 : Préparer les cartes de caractéristiques pour affichage
    feature_maps_resized = []
    titles = []

    for i, fmap in enumerate(feature_maps):
        fmap = fmap[0]  # Retirer la dimension batch

        # **Ignorer les cartes de caractéristiques 1D**
        if fmap.ndim == 1:
            print(f"⚠️ Ignoré : {feature_extractor.output_names[i]} (shape {fmap.shape})")
            continue

        # **Si plusieurs canaux, prendre la moyenne**
        if fmap.ndim == 3 and fmap.shape[-1] > 1:
            fmap = np.mean(fmap, axis=-1)

        feature_maps_resized.append(fmap)
        titles.append(feature_extractor.output_names[i])

    # Vérifier qu'il y a bien des features à afficher
    if len(feature_maps_resized) == 0:
        print("⚠️ Aucune feature map trouvée ! Vérifiez les couches sélectionnées.")
        return

    # Calcul du nombre total de sous-graphes
    total_feature_maps = len(feature_maps_resized)

    # Ajustement dynamique du nombre de lignes
    num_rows = max(1, (total_feature_maps // num_cols) + (1 if total_feature_maps % num_cols else 0))

    # Ajustement de la taille de la figure pour éviter trop d’espacement
    plt.figure(figsize=(num_cols * 2, num_rows * 2))

    # Affichage des cartes de caractéristiques
    for i, (feature_map, title) in enumerate(zip(feature_maps_resized, titles)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(title, fontsize=8)
        plt.imshow(feature_map, cmap="inferno")
        plt.axis('off')

    plt.subplots_adjust(hspace=0.1, wspace=0.1)  # ✅ Réduction de l’espacement
    plt.show()

    # Étape 4 : Calculer le masque prédit
    if feature_maps[-1].ndim == 4 and feature_maps[-1].shape[-1] > 1:  # Vérification si softmax présent
        predicted_mask = np.argmax(feature_maps[-1], axis=-1).squeeze()
    else:
        print("⚠️ Aucune sortie softmax détectée, impossible d'afficher le masque prédit.")
        predicted_mask = None

    # Affichage final : image originale, masque réel et masque prédit
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Masque réel")
    plt.imshow(mask, cmap='inferno')
    plt.axis('off')

    if predicted_mask is not None:
        plt.subplot(1, 3, 3)
        plt.title("Masque prédit")
        plt.imshow(predicted_mask, cmap='inferno')
        plt.axis('off')

    plt.tight_layout()
    plt.show()




##### QUALITE DES PREDICTIONS #################################################

def evaluate_model(model, test_gen):
    """
    Évalue le modèle sur le jeu de test et mesure le temps d'inférence.
    
    Args:
        model : Le modèle TensorFlow/Keras entraîné.
        test_gen : Le générateur de données de test.

    Returns:
        tuple : (temps moyen par image, mean IoU de test, prédictions)
    """
    # Mesurer le temps de prédiction
    start_time = time.time()
    predictions = model.predict(test_gen, verbose=1)
    end_time = time.time()

    # Calculer le temps total et moyen
    total_time = end_time - start_time
    num_images = len(test_gen) * test_gen.batch_size  # Nombre total d'images
    time_per_image = total_time / num_images

    print(f"Temps total d'inférence : {total_time:.4f} secondes")
    print(f"Temps moyen par image : {time_per_image:.4f} secondes")

    # Évaluer le modèle
    test_loss, test_iou = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test meanIoU: {test_iou:.4f}")

    return time_per_image, test_iou, predictions

# ---------------------------------------------------------------------------------------------------#

def plot_confusion_matrix(predictions, test_label_ids_img_paths, class_names):
    """
    Calcule une matrice de confusion multiclasse.

    Arguments:
    - predictions: np.array de dimension (num_samples, height, width, num_classes)
    - test_label_ids_img_paths: Tableau de chemins vers les fichiers d'images des labels (en indices de classe)
    - class_names: liste des noms des classes, taille num_classes

    Retourne:
    - np.array de dimension (num_classes, num_classes) représentant la matrice de confusion
    """
    # Charger les labels depuis les chemins et préparer les dimensions
    num_samples, height, width, num_classes = predictions.shape
    labels = np.zeros((num_samples, height, width), dtype=np.int32)

    for i, label_path in enumerate(test_label_ids_img_paths):
        # Charger l'image des labels en niveaux de gris
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label_image is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {label_path}")

        # Redimensionner l'image pour correspondre aux dimensions des prédictions
        resized_label_image = cv2.resize(label_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Stocker les indices de classe
        labels[i] = resized_label_image

    # Conversion des prédictions one-hot encodées en indices de classe
    pred_indices = np.argmax(predictions, axis=-1)

    # Initialisation de la matrice de confusion
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Remplissage de la matrice de confusion
    for true_class in range(num_classes):
        true_mask = (labels == true_class)

        for pred_class in range(num_classes):
            pred_mask = (pred_indices == pred_class)
            confusion_matrix[true_class, pred_class] += np.logical_and(true_mask, pred_mask).sum()

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="magma_r")
    plt.colorbar(label="Nombre de pixels")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title("Matrice de confusion multiclasse")
    plt.tight_layout()
    plt.show()

    return confusion_matrix

# ---------------------------------------------------------------------------------------------------#

def calculate_iou_per_class(predictions, test_label_ids_img_paths, class_names):
    """
    Calcule l'IoU pour chaque classe sur un ensemble de prédictions et de labels.

    Arguments:
    - predictions: np.array de dimension (num_samples, height, width, num_classes)
    - test_label_ids_img_paths: Liste des chemins vers les fichiers de labels (indices de classe)
    - class_names: Liste des noms des classes, taille num_classes

    Retourne:
    - dict contenant l'IoU pour chaque classe
    """
    # Charger les labels depuis les chemins et préparer les dimensions
    num_samples, height, width, num_classes = predictions.shape
    labels = np.zeros((num_samples, height, width), dtype=np.uint8)

    for i, label_path in enumerate(test_label_ids_img_paths):
        # Charger l'image des labels en niveaux de gris
        label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label_image is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {label_path}")

        # Redimensionner l'image pour correspondre aux dimensions des prédictions
        resized_label_image = cv2.resize(label_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Stocker les indices de classe
        labels[i] = resized_label_image

    # Conversion des prédictions one-hot encodées en indices de classe
    pred_indices = np.argmax(predictions, axis=-1)

    # Initialisation
    iou_dict = {class_name: 0.0 for class_name in class_names}
    iou_counts = {class_name: 0 for class_name in class_names}

    # Calcul des IoU pour chaque classe
    for class_id, class_name in enumerate(class_names):
        # Masques pour la classe actuelle
        true_mask = (labels == class_id)
        pred_mask = (pred_indices == class_id)

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        # Calcul de l'IoU pour la classe actuelle
        if union > 0:
            iou_dict[class_name] = intersection / union

    for class_name, iou in iou_dict.items():
        print(f"Classe {class_name}: IoU = {iou:.4f}")

    return iou_dict