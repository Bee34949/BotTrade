from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# ระบุ API Key ที่ได้จาก Alpha Vantage
api_key = ' LVVID3YC5R0OZ6CB'  # แทนที่ 'YOUR_API_KEY' ด้วย API Key ของคุณ

# สร้างวัตถุ TimeSeries สำหรับดึงข้อมูลการเทรด
ts = TimeSeries(key=api_key, output_format='pandas')

# ดึงข้อมูลการเทรดรายวันของหุ้น AAPL
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# แสดงข้อมูลที่ได้มา
print(data.head())

# บันทึกข้อมูลลงในไฟล์ CSV
data.to_csv('AAPL_data_alpha_vantage.csv')
