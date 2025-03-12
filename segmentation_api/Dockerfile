# Utiliser une image légère avec TensorFlow CPU
FROM tensorflow/tensorflow:2.17.0

# Définir l'environnement de travail
WORKDIR /app

# Copier le fichier de dépendances avant l'installation (optimise le cache Docker)
COPY requirements.txt .

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'intégralité du code source de l'API
COPY . .

# Exposer le port 8000 pour FastAPI
EXPOSE 8000

# Désactiver les logs excessifs de TensorFlow pour alléger les logs API
ENV TF_CPP_MIN_LOG_LEVEL=3

# Lancer l'API FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
