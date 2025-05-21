import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Klasör yolları
RAW_DATA_PATH = "dataset/events.csv"
OUTPUT_DIR = "prepared"

# 1. Veriyi yükle ve zaman damgasını dönüştür
df = pd.read_csv(RAW_DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df.sort_values(["visitorid", "timestamp"])

print("Veriyi yükle ve zaman damgasını dönüştür")
# 2. Event kodlama (view=0, addtocart=1, transaction=2)
event_map = {"view": 0, "addtocart": 1, "transaction": 2}
df["event_code"] = df["event"].map(event_map)
print("")
# 3. 30 dakikalık boşlukla oturum oluştur
df["session_number"] = df.groupby("visitorid")["timestamp"].transform(lambda x: x.diff().gt("30min").cumsum())
df["session_id"] = df["visitorid"].astype(str) + "_" + df["session_number"].astype(str)
print("30 dakikalık boşlukla oturum oluştur")
# 4. Label: Oturumda transaction varsa 1, yoksa 0
session_labels = df.groupby("session_id")["event"].apply(lambda x: int("transaction" in x.values))
print(" Label: Oturumda transaction varsa 1, yoksa 0")
# 5. Input'tan transaction event'lerini çıkar
df_inputs = df[df["event"] != "transaction"]

# 6. Oturumlara göre event_code listesi oluştur
session_sequences = df_inputs.groupby("session_id")["event_code"].apply(list)
print("Oturumlara göre event_code listesi oluştur")
# 7. Padding için optimal maxlen belirleme
lengths = session_sequences.apply(len)
maxlen = int(np.percentile(lengths, 95))  # %95'lik oturum uzunluğu
print(f"Padding için kullanılacak maxlen: {maxlen}")

# 8. Çok uzun oturumları kes ve daha kısa olanları pad et
padded_sequences = pad_sequences(session_sequences.values, maxlen=maxlen, padding="post", truncating="post")

# 9. Label'ları hizala
labels = session_labels.loc[session_sequences.index].values

# 10. Zaman bazlı sıralama ve train/test ayrımı
session_times = df.groupby("session_id")["timestamp"].min()
sorted_sessions = session_times.loc[session_sequences.index].sort_values().index

split_idx = int(len(sorted_sessions) * 0.8)
train_ids = sorted_sessions[:split_idx]
test_ids = sorted_sessions[split_idx:]

X_train = padded_sequences[np.isin(session_sequences.index, train_ids)]
y_train = labels[np.isin(session_sequences.index, train_ids)]
X_test = padded_sequences[np.isin(session_sequences.index, test_ids)]
y_test = labels[np.isin(session_sequences.index, test_ids)]
print("Zaman bazlı sıralama ve train/test ayrımı")
# 11. Kayıt işlemi
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

print("Preprocessing tamamlandı ve dosyalar saved/prepared klasörüne kaydedildi.")
