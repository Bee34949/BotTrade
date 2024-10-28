import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 1. โหลดไฟล์ CSV
data = pd.read_csv('AAPL_data_alpha_vantage.csv')  # ตรวจสอบให้แน่ใจว่าเส้นทางของไฟล์ถูกต้อง

# 2. ทำความสะอาดข้อมูล
data = data.dropna()

# 3. คำนวณค่า Moving Average และ RSI
data['MA_20'] = data['4. close'].rolling(window=20).mean()

def compute_RSI(data, window):
    diff = data.diff(1).dropna()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

data['RSI_14'] = compute_RSI(data['4. close'], 14)
data = data.dropna()

# 4. การเลือกฟีเจอร์และกำหนดเป้าหมาย
features = ['MA_20', 'RSI_14']
target = '4. close'

X = data[features]
y = data[target]

# 5. แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. การสเกลข้อมูลฟีเจอร์และเป้าหมาย
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# สเกลข้อมูลเป้าหมาย (y)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 7. เตรียมข้อมูล sequences สำหรับ LSTM
def create_sequences(data, target, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(target[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 10  # จำนวน timestep ที่ใช้ย้อนหลัง
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, n_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, n_steps)

# 8. ปรับข้อมูลให้มีรูปแบบ (samples, timesteps, features) สำหรับ LSTM
X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], len(features)))
X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], len(features)))

# 9. สร้างโมเดล LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, len(features))))
model.add(LSTM(units=50))
model.add(Dense(1))

# 10. คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# 11. ฝึกสอนโมเดล
model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# 12. บันทึกโมเดลที่ฝึกสอนแล้วเป็นไฟล์ .h5
model.save('models/trading_model.h5')

print("โมเดลได้ถูกฝึกสอนและบันทึกเป็นไฟล์ 'models/trading_model.h5'.")
