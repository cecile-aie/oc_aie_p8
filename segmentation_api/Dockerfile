# Utiliser une image légère avec TensorFlow CPU
FROM tensorflow/tensorflow:2.17.0

# Définir l'environnement de travail
WORKDIR /app

# Copier le fichier de dépendances avant l'installation
COPY requirements.txt .

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Ajouter un utilisateur non-root et lui donner les droits sur /app
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Copier l'intégralité du code source de l'API
COPY --chown=appuser:appuser . .

# Exposer le port 8000 pour FastAPI
EXPOSE 8000

# Désactiver les logs excessifs de TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=3

# Lancer l'API FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
