import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    PrecisionRecallDisplay)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Masking,
    Embedding,
    Bidirectional,
    LSTM,
    GlobalMaxPooling1D,
    Dropout,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


DATA_DIR = Path("prepared")
PADDING_VALUE = -1
VOCAB_SIZE = 3
EMBED_DIM = 16



def load_split(name: str) -> np.ndarray:
    return np.load(DATA_DIR / f"{name}.npy")


def load_data():
    X = load_split("X_train")
    y = load_split("y_train")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_test = load_split("X_test")
    y_test = load_split("y_test")
    seq_len = X_train.shape[1]
    return (X_train, y_train, X_val, y_val, X_test, y_test, seq_len)



def build_s2l_model(seq_len: int) -> tf.keras.Model:
    model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(seq_len,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ],
    )
    return model



def main():
    X_train, y_train, X_val, y_val, X_test, y_test, seq_len = load_data()


    class_weights = {0: 1, 1: 30}  # Önceki: compute_class_weight(...)

    model = build_s2l_model(seq_len)
    es = EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)


    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=1024,
        callbacks=[es],
        class_weight=class_weights,
        verbose=2,
    )

    y_prob = model.predict(X_test, verbose=0).flatten()

    # F1 skoru maksimize eden eşik değeri
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_prob >= best_threshold).astype(np.int8)

    print(f"\n🔧 En İyi Eşik Değeri (F1'e göre): {best_threshold:.4f}")
    print("\nS2L Modeli Performansı:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Eğitim süreci grafiği
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Kayıp (loss) Trendi')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['precision'], label='Eğitim Kesinliği')
    plt.plot(history.history['val_precision'], label='Doğrulama Kesinliği')
    plt.title('Kesinlik (precision) Trendi')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Precision-Recall grafiği
    plt.figure(figsize=(10, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, name="S2L Modeli")
    plt.title("Doğruluk-Duyarlılık Eğrisi", fontsize=13)
    plt.xlabel("Pozitif Öngörü Oranı (Precision)", fontsize=11)
    plt.ylabel("Gerçek Pozitif Oranı (Recall)", fontsize=11)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', label='Baz Model')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ROC Curve grafiği
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}", color="blue", linestyle="-", marker="o")
    plt.plot([0, 1], [0, 1], label="Şans Doğrusu", color="gray", linestyle="--")
    plt.title("ROC Eğrisi (Alıcı İşletim Karakteristik Eğrisi)", fontsize=13)
    plt.xlabel("Yanlış Pozitif Oranı (False Positive Rate)", fontsize=11)
    plt.ylabel("Doğru Pozitif Oranı (True Positive Rate)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()