import MetaTrader5 as mt5
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

# ฟังก์ชันเพื่อโหลดข้อมูลราคาใหม่จาก MetaTrader 5
def get_latest_data(symbol, n=300):  
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n)  
    if rates is None or len(rates) == 0:
        print(f"Failed to get rates for {symbol}, error code: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

# ฟังก์ชันคำนวณตัวบ่งชี้ทางเทคนิค
def add_indicators(data):
    data['MA_20'] = data['close'].rolling(window=20).mean()
    data['RSI_14'] = compute_RSI(data['close'], 14)
    data['MA_50'] = data['close'].rolling(window=50).mean()
    data['MA_200'] = data['close'].rolling(window=200).mean()
    data = data.dropna()  
    return data

# ฟังก์ชันคำนวณ RSI
def compute_RSI(data, window):
    diff = data.diff(1)
    gain = diff.clip(lower=0).rolling(window=window).mean()
    loss = -diff.clip(upper=0).rolling(window=window).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# ฟังก์ชันตรวจจับการกลับตัวของแนวโน้มราคาจาก Moving Average Crossover
def detect_trend_reversal(data):
    if data['MA_50'].iloc[-1] > data['MA_200'].iloc[-1] and data['MA_50'].iloc[-2] <= data['MA_200'].iloc[-2]:
        return 'buy'  
    elif data['MA_50'].iloc[-1] < data['MA_200'].iloc[-1] and data['MA_50'].iloc[-2] >= data['MA_200'].iloc[-2]:
        return 'sell'  
    return 'hold'

# ฟังก์ชันทำนายแนวโน้มตลาด
def predict_trend(model, data):
    if len(data) < 50:
        print("Not enough data to make prediction")
        return 0  

    scaler = MinMaxScaler(feature_range=(0, 1))  
    data = data.reshape(-1, 1)
    
    # ตรวจสอบข้อมูลก่อนทำการสเกล
    print(f"Original data before scaling: {data[-50:]}")

    scaled_data = scaler.fit_transform(data)
    X_test = np.array([scaled_data[-50:]])

    # ทำนายโดยใช้โมเดล
    prediction = model.predict(X_test)
    predicted_value_scaled = prediction[0][0]  # ใช้ค่าทำนายที่ถูกสเกล

    # แปลงกลับจากสเกลมาเป็นค่าจริง
    predicted_value = scaler.inverse_transform(np.array([[predicted_value_scaled]]))[0][0]

    print(f"Predicted value (real): {predicted_value}")  # พิมพ์ค่าทำนายที่แปลงกลับแล้ว
    return predicted_value  # ส่งค่าทำนายที่แปลงกลับแล้ว

# ฟังก์ชันเปิดออเดอร์
def place_order(order_type, symbol, volume=0.1, sl_pips=50, tp_pips=100):
    # ดึงข้อมูลราคาปัจจุบัน
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found, unable to place order.")
        return False

    # ดึงราคา bid/ask
    if order_type == mt5.ORDER_TYPE_BUY:
        price = mt5.symbol_info_tick(symbol).ask
        sl = price - sl_pips * symbol_info.point  # Stop Loss สำหรับ Buy ต่ำกว่าราคาปัจจุบัน
        tp = price + tp_pips * symbol_info.point  # Take Profit สำหรับ Buy สูงกว่าราคาปัจจุบัน
    else:
        price = mt5.symbol_info_tick(symbol).bid
        sl = price + sl_pips * symbol_info.point  # Stop Loss สำหรับ Sell สูงกว่าราคาปัจจุบัน
        tp = price - tp_pips * symbol_info.point  # Take Profit สำหรับ Sell ต่ำกว่าราคาปัจจุบัน

    # สร้างคำสั่งออเดอร์
    order_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,  # ปริมาณล็อตที่ต้องการเปิด
        "type": order_type,
        "price": price,
        "sl": sl,  # Stop Loss
        "tp": tp,  # Take Profit
        "deviation": 10,  # ค่า Slippage ที่ยอมรับได้
        "magic": 123456,  # ID สำหรับระบุคำสั่งนี้
        "comment": "Placed by bot",
        "type_time": mt5.ORDER_TIME_GTC,  # 'Good till Cancelled'
        "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
    }

    # ส่งคำสั่งออเดอร์
    result = mt5.order_send(order_request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed, retcode: {result.retcode}")
        return False
    else:
        print(f"Order placed successfully, retcode: {result.retcode}")
        return True

# ฟังก์ชันปิดออเดอร์
def close_order(ticket):
    order = mt5.positions_get(ticket=ticket)
    if order is None or len(order) == 0:
        print(f"Order {ticket} not found.")
        return False

    position_type = mt5.ORDER_TYPE_SELL if order[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = mt5.symbol_info_tick(order[0].symbol).bid if position_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(order[0].symbol).ask

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": order[0].symbol,
        "volume": order[0].volume,
        "type": position_type,  # ตรงข้ามกับตำแหน่งที่เปิดอยู่
        "position": ticket,
        "price": close_price,
        "deviation": 10,
        "magic": 123456,
        "comment": "Close order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # ส่งคำสั่งปิดออเดอร์
    result = mt5.order_send(close_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close order {ticket}, retcode: {result.retcode}")
        return False
    else:
        print(f"Order {ticket} closed successfully, retcode: {result.retcode}")
        return True

# โหลดโมเดล
try:
    model = load_model('models/trading_model.h5')
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# เชื่อมต่อกับ MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MetaTrader 5, error code =", mt5.last_error())
    exit()

# ตรวจสอบข้อมูลบัญชี
account_info = mt5.account_info()
if account_info is None:
    print("Failed to get account info, error code:", mt5.last_error())
    mt5.shutdown()
    exit()
else:
    print(f"Account ID: {account_info.login}, Balance: {account_info.balance}, Equity: {account_info.equity}")

# เลือกสัญลักษณ์
symbol = 'EURUSD'
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"Symbol {symbol} not found, error code:", mt5.last_error())
    mt5.shutdown()
    exit()
if not symbol_info.visible:
    print(f"Symbol {symbol} is not visible, trying to select...")
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol {symbol}, error code:", mt5.last_error())
        mt5.shutdown()
        exit()

# ทำการเทรดอัตโนมัติตามผลการทำนายและการตรวจจับการกลับตัวของราคา
while True:
    # ดึงข้อมูลราคาล่าสุด
    data_df = get_latest_data(symbol)
    if data_df is None:
        time.sleep(30)  # รอ 30 วินาทีแล้วลองใหม่
        continue

    data_with_indicators = add_indicators(data_df)  

    if data_with_indicators.empty or len(data_with_indicators) < 50:
        print("Not enough data for indicators")
        time.sleep(30)  # รอ 30 วินาทีแล้วลองใหม่
        continue

    # ทำนายแนวโน้มตลาด
    predicted_value = predict_trend(model, data_with_indicators['close'].values)

    # ตรวจจับการกลับตัวของราคา
    trend_reversal = detect_trend_reversal(data_with_indicators)
    print(f"Trend Reversal: {trend_reversal}")  # พิมพ์ค่า Trend Reversal

    try:
        current_rsi = data_with_indicators['RSI_14'].iloc[-1]
        print(f"RSI: {current_rsi}")  # พิมพ์ค่า RSI
    except IndexError:
        print("Not enough data to compute RSI")
        time.sleep(30)  # รอ 30 วินาทีแล้วลองใหม่
        continue

    # เงื่อนไขการซื้อขายที่ปรับปรุงใหม่
    current_price = data_with_indicators['close'].iloc[-1]  # ใช้ราคาปัจจุบันของสินทรัพย์

    # ตรวจสอบตำแหน่งที่เปิดอยู่และปิดอัตโนมัติหากถึง SL หรือ TP
    open_positions = mt5.positions_get(symbol=symbol)
    if open_positions:
        for pos in open_positions:
            ticket = pos.ticket
            if pos.profit >= 50 or pos.profit <= -20:  # ตัวอย่างการตรวจสอบกำไร/ขาดทุน
                print(f"Closing order {ticket} with profit: {pos.profit}")
                close_order(ticket)

    # กำหนด threshold สำหรับการเปลี่ยนแปลงราคา
    threshold = 0.01  # ค่า threshold ที่ยอมรับได้สำหรับการเปลี่ยนแปลงราคา
    price_diff = abs(predicted_value - current_price)

    max_open_positions = 50  # จำกัดจำนวนออเดอร์ที่เปิดได้สูงสุด
    if len(open_positions) < max_open_positions:
        if price_diff > threshold:
            if predicted_value > current_price and current_rsi < 70 and (trend_reversal == 'buy' or trend_reversal == 'hold'):
                print("Buy condition met. Placing buy order.")
                success = place_order(mt5.ORDER_TYPE_BUY, symbol, sl_pips=50, tp_pips=100)
            elif predicted_value < current_price and current_rsi > 30 and (trend_reversal == 'sell' or trend_reversal == 'hold'):
                print("Sell condition met. Placing sell order.")
                success = place_order(mt5.ORDER_TYPE_SELL, symbol, sl_pips=50, tp_pips=100)
        else:
            print(f"Price difference too small: {price_diff}. No trade action.")
    else:
        print("Max open positions reached. No new orders will be placed.")



    # รอ 30 วินาที ก่อนทำงานรอบถัดไป
    time.sleep(30)