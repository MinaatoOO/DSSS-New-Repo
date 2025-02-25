import matplotlib
matplotlib.use('Agg')  # Utilisation du backend pour les environnements sans GUI
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf

# ================================
# ✅ 1️⃣ Afficher la version de TensorFlow
# ================================
print("TensorFlow version:", tf.__version__)

# ================================
# ✅ 2️⃣ Créer le dossier pour les résultats
# ================================
output_dir = "test_CNN"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Dossier '{output_dir}' créé pour stocker les résultats.")

# ================================
# ✅ 3️⃣ Charger le modèle
# ================================
model = load_model("dsss_cnn_model.h5", compile=False)
print("✅ Modèle chargé avec succès.")

# ================================
# 📁 4️⃣ Charger les données de test
# ================================
X_test = np.load("dataset_dsss/X.npy")
y_test = np.load("dataset_dsss/y.npy")

# ================================
# 🏷️ 5️⃣ Évaluer le modèle
# ================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"📊 Précision sur les données de test : {accuracy * 100:.2f}%")
print(f"📉 Perte sur les données de test : {loss:.4f}")

# ================================
# 🤔 6️⃣ Prédictions et Évaluation
# ================================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# ================================
# 📊 7️⃣ Matrice de confusion
# ================================
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bruit", "DSSS"])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de Confusion")
plt.savefig(os.path.join(output_dir, "matrice_confusion.png"))
plt.close()
print("✅ Matrice de confusion enregistrée.")

# ================================
# 📊 8️⃣ Rapport de classification
# ================================
report = classification_report(y_true_classes, y_pred_classes, target_names=["Bruit", "DSSS"])
print("\n📋 Rapport de classification :")
print(report)

# Sauvegarder le rapport dans un fichier texte
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)
print("✅ Rapport de classification sauvegardé.")

# ================================
# 📊 9️⃣ Visualiser quelques prédictions
# ================================
num_samples = 5  # Nombre d'exemples à afficher
indices = np.random.choice(len(X_test), num_samples, replace=False)

for idx in indices:
    signal = X_test[idx].reshape(-1, 2)  # Reformatage si nécessaire
    plt.figure(figsize=(10, 4))
    plt.plot(signal[:, 0], label="I (In-phase)")
    plt.plot(signal[:, 1], label="Q (Quadrature)")
    plt.title(f"Vrai label : {'DSSS' if y_true_classes[idx] == 1 else 'Bruit'} | Prédit : {'DSSS' if y_pred_classes[idx] == 1 else 'Bruit'}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"prediction_{idx}.png"))
    plt.close()
print("✅ Graphiques des prédictions sauvegardés.")

# ================================
# 📊 🔟 Courbes d'entraînement depuis training_history.pkl
# ================================
try:
    with open("training_history.pkl", 'rb') as f:
        history = pickle.load(f)
    
    # Courbe de précision
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Précision Entraînement')
    plt.plot(history['val_accuracy'], label='Précision Validation')
    plt.title('Courbe de Précision')
    plt.xlabel('Époques')
    plt.ylabel('Précision')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "courbe_precision.png"))
    plt.close()
    print("✅ Courbe de précision enregistrée.")

    # Courbe de perte
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Perte Entraînement')
    plt.plot(history['val_loss'], label='Perte Validation')
    plt.title('Courbe de Perte')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "courbe_perte.png"))
    plt.close()
    print("✅ Courbe de perte enregistrée.")

except FileNotFoundError:
    print("⚠️ Fichier 'training_history.pkl' non trouvé. Courbes d'entraînement non générées.")

# ================================
# ✅ Fin du script
# ================================
print(f"\n📁 Tous les résultats sont sauvegardés dans le dossier '{output_dir}'.")
