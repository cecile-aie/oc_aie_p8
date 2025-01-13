# Utiliser l'image officielle TensorFlow avec GPU et Jupyter Lab
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

# Définir l'environnement de travail
WORKDIR /tf

# Installer les bibliothèques nécessaires
RUN pip install --no-cache-dir pyquaternion==0.9.9 albumentations==2.0.0 segmentation_models==1.0.1 pandas==2.2.3

# Exposer le port pour Jupyter Lab
EXPOSE 8888

# Lancer Jupyter Lab lorsque le conteneur démarre
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
