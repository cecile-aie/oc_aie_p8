import tensorflow as tf
import os  # <-- Ajout de l'import

# 🔹 Forcer TensorFlow à utiliser uniquement le CPU
tf.config.set_visible_devices([], 'GPU')

# 🔹 Définir le chemin du modèle (compatible avec le conteneur Docker)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/segmentation_model.keras")

# 🔹 Charger le modèle de segmentation
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 🔹 Dictionnaire des classes avec leurs couleurs associées
CLASS_INFO = {
    0: {"name": "void", "color": (0, 0, 0)},        # Noir
    1: {"name": "flat", "color": (255, 153, 0)},      # Orange
    2: {"name": "construction", "color": (128, 128, 128)},  # Jaune
    3: {"name": "object", "color": (0, 0, 255)},    # Bleu
    4: {"name": "nature", "color": (0, 255, 0)},  # Vert
    5: {"name": "sky", "color": (0, 255, 255)},     # Cyan
    6: {"name": "human", "color": (255, 0, 255)},   # Magenta
    7: {"name": "vehicle", "color": (255, 0, 0)}  # Rouge
}

# 🔹 Liste ordonnée des noms de classes (utile pour l’API)
CLASS_NAMES = [info["name"] for info in CLASS_INFO.values()]

# 🔹 Liste des couleurs sous forme de tableau (utile pour le post-processing)
CLASS_COLORS = [info["color"] for info in CLASS_INFO.values()]
