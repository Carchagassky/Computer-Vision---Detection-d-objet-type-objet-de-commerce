# Computer-Vision---Detection-d-objet-type-objet-de-commerce

Projet de cours Computer Vision.

Objectif : faire de la détection d'objets que l'on peut retrouver dans le commerce.

contexte: C’est un projet sur la classification des objets que l’on pourrait vendre. On pourrait les ranger par type d’éléments, et analyser ce qui partirait en premier pour une mise en rayon plus efficace ou pour un contrôle des stocks plus rapide.


la motivation: C’est un thème d’actualité, surtout en lien avec la reconnaissance d’objets


les données disponibles pour réaliser le projet: Dataset COCO, bibliothèque tensorflow ; matplotlib; opencv


les idées de modèles pour le réaliser: Model U-Net; Modele Mask-CNN; DeepLab v3+  etc…


le modèle sélectionné et la raison de sélection: Model U-Net, car idéal pour la suppression d’arrière plan pour les images de produits standardisés.


le prévisionnel des tâches à réaliser :
1 -réaliser l'importation des données  	
2 -classification des données ( test, entraînement, données finales)  
3 -entrainement du modèle	
4 -Raffinement du modèle		


Il faut importer et extraire dans la même racine le dataset trouvable ici : https://github.com/marcusklasson/GroceryStoreDataset
