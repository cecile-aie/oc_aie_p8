# Utiliser une image légère de Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances et installer les packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy

# Log du commit actuel
ARG GIT_COMMIT
RUN echo $GIT_COMMIT > /app/VERSION.txt

# Copier le reste du code de la webapp
COPY . .

# Exposer le port 5000 pour Flask
EXPOSE 5000

# Définir la variable d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Lancer l'application Flask
CMD ["flask", "run"]
