import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle  # Pour sauvegarder l'historique


# ✅ Vérifier la disponibilité des GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU détecté : {gpus[0].name}")
    try:
        # Autoriser la croissance de la mémoire GPU pour éviter les erreurs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ Aucun GPU détecté. Utilisation du CPUUUUUU")



# ================================
# 📁 1️⃣ Chargement des données
# ================================
def load_dataset(data_path="dataset_dsss"):
    X = np.load(f"{data_path}/X.npy")  # Forme attendue : (22000, 2048, 1, 2)
    y = np.load(f"{data_path}/y.npy")  # Labels : 0 (bruit) ou 1 (DSSS)
    y = to_categorical(y, num_classes=2)  # Conversion en one-hot encoding
    return X, y

# ================================
# 🏗️ 2️⃣ Construction du CNN (ResNet-like)
# ================================
def residual_block(x, filters, kernel_size=3, stride=1):
    """
    Implémente un bloc résiduel (ResNet Block)
    """
    shortcut = x  # Connexion résiduelle
    
    # Première convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Deuxième convolution
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Si changement de dimensions, ajuster le shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Ajout du shortcut
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def build_cnn(input_shape=(2048, 1, 2)):
    """
    Construction du CNN pour la détection DSSS
    """
    inputs = layers.Input(shape=input_shape)

    # Normalisation (conformément à la formule (14) du document)
    x = layers.Lambda(lambda x: x / (tf.reduce_mean(tf.square(tf.abs(x)), axis=1, keepdims=True) + 1e-10))(inputs)

    # Première couche de convolution
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 8 Blocs Résiduels
    for i in range(8):
        stride = 2 if i % 2 == 0 else 1  # Réduction des dimensions tous les 2 blocs
        x = residual_block(x, filters=32 * (2 ** (i // 2)), stride=stride)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Couche dense finale
    outputs = layers.Dense(2, activation='softmax')(x)  # 2 classes : DSSS ou bruit

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ================================
# 🏋️ 3️⃣ Entraînement du modèle
# ================================
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks pour EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entraînement
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop])

    # ✅ Sauvegarder l'historique après l'entraînement
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("✅ Historique d'entraînement sauvegardé sous 'training_history.pkl'")

    return history

# ================================
# 📊 4️⃣ Visualisation des courbes
# ================================
def plot_training(history):
    # Crée un dossier 'plots' s'il n'existe pas
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure(figsize=(12, 5))

    # Courbe de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entraînement')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision')
    plt.xlabel('Époques')
    plt.ylabel('Précision')
    plt.legend()

    # Sauvegarde la courbe de précision
    precision_path = os.path.join(plot_dir, 'precision_plot.png')
    plt.savefig(precision_path)
    print(f"✅ Courbe de précision enregistrée dans : {precision_path}")

    plt.figure(figsize=(12, 5))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Entraînement')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Fonction de perte')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()

    # Sauvegarde la courbe de perte
    loss_path = os.path.join(plot_dir, 'loss_plot.png')
    plt.savefig(loss_path)
    print(f"✅ Courbe de perte enregistrée dans : {loss_path}")

    # Fermer les figures pour libérer la mémoire
    plt.close('all')
# ================================
# 🚀 5️⃣ Lancement du programme
# ================================
if __name__ == "__main__":
    # 📁 Chargement des données
    X, y = load_dataset("dataset_dsss")

    # ✂️ Division en Train/Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🏗️ Construction du modèle
    model = build_cnn(input_shape=(2048, 1, 2))
    model.summary()

    # 🏋️ Entraînement du modèle
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 📊 Visualisation des résultats
    plot_training(history)

    # 💾 Sauvegarde du modèle
    model.save("dsss_cnn_model.h5")
    print("✅ Modèle sauvegardé sous 'dsss_cnn_model.h5'")
