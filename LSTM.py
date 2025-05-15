import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras import mixed_precision

# 0. Mixed Precision Ayarı
#mixed_precision.set_global_policy('mixed_float16')

# 1. Veri Yükleme
events = pd.read_csv("dataset/events.csv")
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
events = events.sort_values(['visitorid', 'timestamp'])

# 2. Oturum Belirleme (Optimize edilmiş versiyon)
events['session_number'] = (
    events.groupby('visitorid')['timestamp']
    .transform(lambda x: x.diff().gt('30min').cumsum())
)
events['session_id'] = events['visitorid'].astype(str) + "_" + events['session_number'].astype(str)

# 3. Event Kodlama
event_mapping = {'view': 0, 'addtocart': 1, 'transaction': 2}
events['event_code'] = events['event'].map(event_mapping)
sequences = events.groupby('session_id')['event_code'].apply(list)
labels = events.groupby('session_id').apply(lambda x: 1 if x['event'].iloc[-1] == 'transaction' else 0)

# 4. Padding ve Split
X = pad_sequences(sequences, padding='pre', maxlen=30).astype(np.int32)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. Class Weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(weights)}

# 6. Model Tanımı
model = Sequential([
    Embedding(input_dim=3, output_dim=4, input_length=30, mask_zero=True),
    LSTM(8),
    Dense(1, activation='sigmoid', dtype='float32')  # float16 yerine float32 çıkış
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

# 7. EarlyStopping
early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)

# 8. Eğitim
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=15,
    batch_size=1024,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# 9. Değerlendirme
y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

print("## PERFORMANCE REPORT ##")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_probs))
