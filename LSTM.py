import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from sklearn.metrics import classification_report, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt

# GPU kontrolü ve bellek optimizasyonu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU kullanılıyor: {gpus[0].name}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("❌ GPU bulunamadı.")

# Veri yükleme
X_train = np.load("prepared/X_train.npy")
y_train = np.load("prepared/y_train.npy")
X_test = np.load("prepared/X_test.npy")
y_test = np.load("prepared/y_test.npy")

# Model Mimarisi
model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Embedding(input_dim=3, output_dim=16),
    LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    Dropout(0.4),
    LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Optimize edilmiş hiperparametreler
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    clipnorm=1.0
)

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()

# Sınıf ağırlıklarını hesapla
class_weights = {0: 1, 1: 20}  # Pozitif sınıfa 20x ağırlık
print(f"\n⚠️ Sınıf Ağırlıkları: {class_weights}")

# Early Stopping (Precision odaklı)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_precision',
    patience=5,
    mode='max',
    restore_best_weights=True
)

# Model eğitimi
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=1024,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=2
)

# Threshold optimizasyonu
y_pred_probs = model.predict(X_test).flatten()
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_probs > best_threshold).astype(int)

# Detaylı performans raporu
print("\n📊 Detaylı Performans Raporu:")
print(classification_report(y_test, y_pred, digits=4))

# Precision-Recall Eğrisi
plt.figure(figsize=(10, 6))
PrecisionRecallDisplay.from_predictions(y_test, y_pred_probs, name="LSTM Modeli")
plt.title("Doğruluk-Duyarlılık Eğrisi", fontsize=13)
plt.xlabel("Pozitif Öngörü Oranı (Precision)", fontsize=11)
plt.ylabel("Gerçek Pozitif Oranı (Recall)", fontsize=11)
plt.plot([0, 1], [0.5, 0.5], linestyle='--', label='Baz Model')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.show()

# Eğitim Geçmişi Görselleştirme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp (loss) Trendi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['precision'], label='Eğitim Kesinliği')
plt.plot(history.history['val_precision'], label='Doğrulama Kesinliği')
plt.title('Kesinlik (presicion) Trendi')
plt.legend()

plt.tight_layout()
plt.show()