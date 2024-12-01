Project Overview
This project aims to develop an Automated Trading Bot that executes trades based on technical indicators like Moving Average (MA) and Relative Strength Index (RSI) using MetaTrader 5 (MT5). The bot combines Python for signal generation via AI models and MQL5 for executing trades directly within the MT5 platform.

Key Features
Python Integration:

Fetch historical price data from MT5.
Analyze and generate trading signals using AI models trained with TensorFlow/Keras.
Place and close orders with precise parameters like Stop Loss (SL) and Take Profit (TP).
MQL5 Expert Advisor:

Respond to signals based on MA and RSI for automated execution.
Operates within the MetaTrader 5 trading environment.
Components and Tools
MetaTrader 5: For executing trade orders and backtesting strategies.
Python: For AI-based signal generation.
TensorFlow/Keras: Framework for building and running AI prediction models.
Technical Indicators:
Moving Average (MA): Trend identification.
Relative Strength Index (RSI): Identifies overbought/oversold conditions.
Code Structure
Python Code
Data Processing:

Fetch historical data using get_latest_data().
Calculate MA and RSI using add_indicators().
AI Predictions:

Load pre-trained AI model (.h5 file).
Predict market trends with predict_trend().
Order Execution:

Use place_order() and close_order() to interact with MT5.
MQL5 Code
Indicator Setup:

Initialize MA and RSI in OnInit().
Tick-based Actions:

Evaluate buy/sell conditions in OnTick().
Execute trades when conditions are met.
Trade Management:

Ensure trades align with account permissions using AccountInfoInteger().
Installation and Usage
MetaTrader 5 (MQL5 Setup)
Open MetaEditor, create a new .mq5 file, and copy the provided MQL5 code.
Compile and add the Expert Advisor (EA) to your MT5 terminal.
Python Script
Install required libraries: MetaTrader5, TensorFlow, pandas.
Run the script:
bash
Copy code
python <filename>.py
Verify logs for signal generation and order execution.
Strategy Tester Usage
Open Strategy Tester in MT5.
Select your EA, set the time period and trading pair.
Start backtesting and analyze performance metrics.
Troubleshooting
Connection Issues: Ensure proper internet connectivity and MT5 server status.
Order Errors:
10027: Check broker conditions like lot size.
10018: Market closed; verify trading hours.
Caution
Always backtest your strategy before deploying in a live account.
Adjust parameters (e.g., SL/TP, RSI thresholds) to optimize performance.
Flowchart
text
Copy code
Start -> Connect to MetaTrader5 -> Fetch Historical Data -> Calculate Indicators ->
Predict Market Trends -> Check Buy/Sell Conditions -> Execute Order ->
Monitor Open Trades -> Repeat
Future Enhancements
Optimize AI models with additional data.
Integrate new indicators and strategies.
Enhance logging and error handling for scalability.
