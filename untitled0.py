# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:12:59 2024

@author: Xavie
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
import os

# Charger le fichier CSV
csv_path = "grocery_store_dataset.csv"
data = pd.read_csv(csv_path)

# Afficher les premières lignes
print("Aperçu des données :")
print(data.head())

# Visualiser quelques exemples d'images
def visualize_examples(data, num_examples=5):
    examples = data.sample(num_examples)  # Sélectionner des exemples aléatoires
    plt.figure(figsize=(15, 10))
    for i, row in enumerate(examples.itertuples()):
        image_path = row.image_path
        label = row.label
        # Charger l'image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Afficher
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.show()

# Visualiser quelques exemples d'images
visualize_examples(data, num_examples=5)


#Prétraitement
# Redimensionner et normaliser les images
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normaliser entre 0 et 1
    return img

# Prétraiter les images et labels
def prepare_dataset(data, target_size=(224, 224)):
    images = []
    labels = []
    label_to_index = {label: idx for idx, label in enumerate(data['label'].unique())}
    for row in data.itertuples():
        img = preprocess_image(row.image_path, target_size=target_size)
        images.append(img)
        labels.append(label_to_index[row.label])
    return np.array(images), np.array(labels), label_to_index

# Préparer le dataset complet
images, labels, label_to_index = prepare_dataset(data)
print(f"Images shape : {images.shape}")
print(f"Labels shape : {labels.shape}")
print(f"Mapping des labels : {label_to_index}")


#Division en ensembles d'entraînement, validation et test
#Actions principales :
#Divisez les données en :
#Entraînement (70%)
#Validation (20%)
#Test (10%)

# Diviser les données en train, validation et test
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.3, random_state=42
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.33, random_state=42
)

print(f"Train : {train_images.shape}, {train_labels.shape}")
print(f"Validation : {val_images.shape}, {val_labels.shape}")
print(f"Test : {test_images.shape}, {test_labels.shape}")


#Construire le modèle
#Choix d'un modèle pré-entraîné :
#Nous utiliserons un modèle comme ResNet50 pour une classification rapide.


# Définir le modèle avec ResNet50
model = Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_to_index), activation='softmax')  # Nombre de classes
])

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Afficher le résumé du modèle
model.summary()


# Entraîner le modèle
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=10,  # Nombre d'époques
    batch_size=32,  # Taille des lots
    verbose=1
)

#Visualisation de l'entraînement
#Courbes de précision et de perte
#Le suivi de la précision (accuracy) et de la perte (loss) pendant l'entraînement permet d’évaluer la convergence du modèle.

# Visualiser les performances d'entraînement
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Afficher les courbes
plot_training_history(history)


# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Visualisation des Prédire des classes sur l'ensemble de test
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Visualiser quelques prédictions
def visualize_predictions(images, true_labels, predicted_labels, label_to_index, num_examples=5):
    index_to_label = {v: k for k, v in label_to_index.items()}
    plt.figure(figsize=(15, 10))
    for i in range(num_examples):
        img = images[i]
        true_label = index_to_label[true_labels[i]]
        predicted_label = index_to_label[predicted_labels[i]]

        plt.subplot(1, num_examples, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
        plt.axis("off")
    plt.show()

# Afficher les prédictions
visualize_predictions(test_images, test_labels, predicted_labels, label_to_index)
