# Utiliser l'image officielle TensorFlow avec GPU et Jupyter Lab
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

# Définir l'environnement de travail
WORKDIR /tf

# Installer les bibliothèques nécessaires
RUN apt update && apt install -y graphviz \
    && pip install --no-cache-dir pyquaternion==0.9.9 albumentations==2.0.4 segmentation_models==1.0.1 pandas==2.2.3 scikit-learn==1.6.1 pydot

# Exposer les ports pour Jupyter Lab et TensorBoard
EXPOSE 8888 6006

# Définir les variables d’environnement pour éviter le warning NUMA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV JUPYTER_ENABLE_LAB=yes
ENV TF_NUMA_DISABLED=1
ENV NO_ALBUMENTATIONS_UPDATE=1

# Lancer uniquement Jupyter Lab au démarrage du conteneur
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

