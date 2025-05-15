import pandas as pd
import numpy as np

events = pd.read_csv("dataset/events.csv")
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

events_sorted = events.sort_values(by=['visitorid', 'timestamp'])

events_sorted['session_id'] = (
    (events_sorted.groupby('visitorid')['timestamp'].diff() >= pd.Timedelta(minutes=30))
    .fillna(True)
    .astype(int)
    .groupby(events_sorted['visitorid']).cumsum()
    .astype(str)
)

events_sorted['session_id'] = (
    events_sorted['visitorid'].astype(str) + "_" + events_sorted['session_id']
)

session_labels = events_sorted.groupby('session_id')['event'].apply(lambda x: int('transaction' in x.values))
events_sorted = events_sorted.merge(session_labels.rename("purchase"), on="session_id")

#print(events_sorted[['session_id', 'purchase']].drop_duplicates()['purchase'].value_counts())

#FEATURE ENGINEERING

events_filtered = events_sorted[events_sorted['event'].isin(['view', 'addtocart'])]

# Yeni total_events
total_events = events_filtered.groupby('session_id').size().rename('total_events')

# Diğer feature'lar da bu events_filtered üzerinden hesaplanmalı (data leak'e engel olmak için)
# 2. View event sayısı
num_views = events_sorted[events_sorted['event'] == 'view'].groupby('session_id').size().rename('num_views')

# 3. Addtocart event sayısı
num_addtocarts = events_sorted[events_sorted['event'] == 'addtocart'].groupby('session_id').size().rename('num_addtocarts')

# 4. Unique item sayısı (kaç farklı ürün görüldü)
unique_items_viewed = events_sorted.groupby('session_id')['itemid'].nunique().rename('unique_items_viewed')

# 5. Oturum süresi (son - ilk timestamp, saniye cinsinden)
session_duration = events_filtered.groupby('session_id')['timestamp'].agg(
    lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0
).rename('session_duration')

# 6. View to cart ratio
view_to_cart_ratio = (num_addtocarts / num_views).replace([np.inf, np.nan], 0).rename('view_to_cart_ratio')

# 7. Cart occurred (binary 0/1 olarak sepete ürün eklenmiş mi?)
cart_occurred = (num_addtocarts > 0).astype(int).rename('cart_occurred')

# 8. Last event type (oturumun sonundaki event tipi)
last_event_type = events_sorted.groupby('session_id')['event'].last().rename('last_event_type')

# 9. Hedef değişken (purchase)
purchase = events_sorted.drop_duplicates('session_id')[['session_id', 'purchase']].set_index('session_id')['purchase']

features = pd.concat([
    total_events,
    num_views,
    num_addtocarts,
    unique_items_viewed,
    session_duration,
    view_to_cart_ratio,
    cart_occurred,
    last_event_type,
    purchase
], axis=1).fillna(0)


features['num_views'] = features['num_views'].astype(int)
features['num_addtocarts'] = features['num_addtocarts'].astype(int)
features['unique_items_viewed'] = features['unique_items_viewed'].astype(int)
features['cart_occurred'] = features['cart_occurred'].astype(int)
features['purchase'] = features['purchase'].astype(int)



features.to_csv("dataset/features.csv", index=True)
