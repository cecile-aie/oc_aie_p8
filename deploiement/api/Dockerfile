# Utiliser une image légère avec TensorFlow CPU
FROM tensorflow/tensorflow:2.17.0

# Définir l'environnement de travail
WORKDIR /app

# Copier et installer les dépendances avant d'ajouter le code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pour les tests de charge (peut être supprimé en prod)
RUN pip install --no-cache-dir --ignore-installed blinker locust

# Log du commit
ARG GIT_COMMIT
RUN echo $GIT_COMMIT > /app/version.txt

# 🔹 Ajouter un utilisateur non-root et lui donner les droits sur /app
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Copier uniquement le code source après l’installation des dépendances
COPY --chown=appuser:appuser . .

# 🔹 Exposer le port 8000 pour FastAPI
EXPOSE 8000

# 🔹 Désactiver les logs excessifs de TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=3

# 🔹 Lancer FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
