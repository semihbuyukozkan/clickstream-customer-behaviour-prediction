import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ EVENTS.CSV ANALİZİ -----------------

events = pd.read_csv("dataset/events.csv")

events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms') #timestamp'ı datetime formatına çevirmek için

#TEMEL ANALİZLER (Gereken analizler commentten çıkartılıp çalıştırılabilir..)
'''
print(events.head()) #ilk 5 satıra göz atmak için

print(events.info()) #veri yapısını kontrol etmek için

print(f"Toplam event sayısı: {len(events)}")

print(f"Benzersiz visitorid sayısı: {events['visitorid'].nunique()}")

print(f"Benzersiz itemid sayısı: {events['itemid'].nunique()}")

print("Event türü dağılımı:")
print(events['event'].value_counts())

print(f"En erken event zamanı: {events['timestamp'].min()}")
print(f"En geç event zamanı: {events['timestamp'].max()}")
'''
'''
#transactionid anomalisi kontrolü
anomalies = events[~events['transactionid'].isna() & (events['event'] != 'transaction')]
print("Transaction olmayan ama transactionid içeren satır sayısı:", len(anomalies))
print(anomalies.head())

#visitor başına event sayısı dağılımı
event_counts = events.groupby("visitorid").size()
print("Ziyaretçi başına event sayısı istatistikleri:")
print(event_counts.describe())

#eksik veri kontrolü
print("Eksik veri kontrolü:")
print(events.isna().sum())
'''

#DAVRANIŞ TÜRLERİNİN ZAMAN İÇİNDEKİ DAĞILIMI
'''
# Günlük event sayısını hesapla
events['date'] = events['timestamp'].dt.date
daily_events = events.groupby(['date', 'event']).size().unstack(fill_value=0)

# Grafik çizimi
plt.figure(figsize=(15,6))

plt.plot(daily_events.index, daily_events['view'], label='view', color='green', linestyle='-', marker='.')
plt.plot(daily_events.index, daily_events['addtocart'], label='addtocart', color='blue', linestyle='--')
plt.plot(daily_events.index, daily_events['transaction'], label='transaction', color='red', linestyle=':')

plt.title('Davranış Türlerinin Günlük Zaman Dağılımı')
plt.xlabel('Tarih')
plt.ylabel('Etkileşim Sayısı')
plt.grid(True)
plt.legend(title='Etkileşim Türü')
plt.tight_layout()
plt.show()
'''

#ZİYARETÇİ DAVRANIŞ ZİNCİRLERİ
'''
# Ziyaretçi bazlı zaman sıralı event zinciri
events_sorted = events.sort_values(by=['visitorid', 'timestamp'])

# Ziyaretçi başına sıralı event zinciri (tekrarsız)
user_event_sequences = (
    events_sorted
    .groupby('visitorid')['event']
    .apply(lambda x: ','.join(pd.Series(x).drop_duplicates()))
)

# Kombinasyonların frekansını al
pattern_counts = user_event_sequences.value_counts()

print("En yaygın kullanıcı davranış zincirleri:")
print(pattern_counts)
'''

#EN POPÜLER ÜRÜNLERİN ANALİZİ
'''
# Her event türü için en çok işlem gören ürünleri hesapla
top_viewed = events[events['event'] == 'view']['itemid'].value_counts().head(10)
top_added = events[events['event'] == 'addtocart']['itemid'].value_counts().head(10)
top_bought = events[events['event'] == 'transaction']['itemid'].value_counts().head(10)

print("🔹 En çok görüntülenen ürünler:\n", top_viewed)
print("\n🔹 En çok sepete eklenen ürünler:\n", top_added)
print("\n🔹 En çok satın alınan ürünler:\n", top_bought)
'''

#VERİYİ OTURUM BAZLI HALE GETİRME
'''
# Ziyaretçi bazında sırala
events_sorted = events.sort_values(by=['visitorid', 'timestamp'])

# Zaman farkı (aynı visitor içinde)
events_sorted['time_diff'] = events_sorted.groupby('visitorid')['timestamp'].diff().dt.total_seconds()

# Yeni oturumlar için işaretleyici
session_threshold = 30 * 60  # 30 dakika
events_sorted['new_session'] = (events_sorted['time_diff'] > session_threshold) | (events_sorted['time_diff'].isna())

# Kümülatif oturum numarası üret
events_sorted['session_id'] = events_sorted.groupby('visitorid')['new_session'].cumsum()

# Her ziyaretçi için oturum ID'sini benzersizleştir
events_sorted['session_id'] = events_sorted['visitorid'].astype(str) + '_' + events_sorted['session_id'].astype(str)

# Örnek ziyaretçiden birkaç tanesinin eventlerini kontrol edelim
sample_visitor = events_sorted['visitorid'].sample(1).iloc[0]

sample_session = events_sorted[events_sorted['visitorid'] == sample_visitor][['timestamp', 'event', 'session_id']]
print(sample_session)

session_lengths = events_sorted.groupby('session_id').size()
print(session_lengths.describe())
'''
'''
# Ziyaretçiye ve zamana göre sırala
events_sorted = events.sort_values(by=['visitorid', 'timestamp'])

events_sorted['session_id'] = (
    # Aynı kullanıcı içinde 30 dakikadan uzun boşlukları yeni oturum olarak işaretle
    (events_sorted.groupby('visitorid')['timestamp'].diff() >= pd.Timedelta(minutes=30))
    .fillna(True)  # İlk etkinlikler için True (yeni oturum)
    .astype(int)   # True/False'u 1/0'a çevir
    .groupby(events_sorted['visitorid']).cumsum()  # Kullanıcı içinde artırımlı oturum numarası
    .astype(str)   # Stringe çevir (session_id'nin son parçası olarak)
)

events_sorted['session_id'] = (
    events_sorted['visitorid'].astype(str) + "_" + events_sorted['session_id']
)

print(events_sorted[['visitorid', 'timestamp', 'session_id']].head(10))
'''