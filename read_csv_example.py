import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LSTM  # type: ignore

# 1. อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('AAPL_data_alpha_vantage.csv')

# 2. การทำความสะอาดข้อมูล (Data Cleaning)
# ลบแถวที่มีข้อมูลที่ขาดหาย (NaN)
data = data.dropna()

# 3. การแปลงข้อมูล (Data Transformation)
# คำนวณ Moving Average (MA) 20 วัน
data['MA_20'] = data['4. close'].rolling(window=20).mean()

# คำนวณ RSI 14 วัน
def compute_RSI(data, window):
    diff = data.diff(1).dropna()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

data['RSI_14'] = compute_RSI(data['4. close'], 14)

# ลบแถวที่มีข้อมูลคำนวณที่เป็น NaN หลังจากสร้าง Feature
data = data.dropna()

# 4. การเลือกคุณสมบัติ (Feature Selection) และเตรียมข้อมูลการเทรน
features = ['MA_20', 'RSI_14']  # เลือก Features
target = '4. close'  # เป้าหมายคือราคาปิด

# สร้างชุดข้อมูล Features (X) และ Target (y)
X = data[features]
y = data[target]

# 5. แบ่งข้อมูลเป็นชุดการฝึกสอน (Training Set) และทดสอบ (Test Set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. การสเกลข้อมูล (Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. ฟังก์ชันสร้างลำดับข้อมูลสำหรับ LSTM
def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# กำหนดจำนวน time steps
time_steps = 10

# การเตรียมข้อมูลลำดับเวลา (sequences) สำหรับ LSTM
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, time_steps)

# 8. สร้างและฝึกสอนโมเดล Machine Learning (LSTM)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mean_squared_error')

# ฝึกสอนโมเดล
model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# 9. บันทึกโมเดลที่ฝึกสอนแล้ว
model.save('trading_model.h5')

print("Model has been trained and saved as 'trading_model.h5'.")
