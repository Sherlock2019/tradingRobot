

# AITRADE : the Tradingbot using ML 

# In this example, we've done the following:

# Fetched historical price data.
# Calculated the MACD indicator, which is a common technical analysis tool.
# Created a function to label the data with buy or sell signals based on the MACD.
# Split the data into training and test sets.
# Initialized a Random Forest classifier, trained it on the training set, and evaluated it on the test set.
# Please remember that this is a very basic example and does not account for many factors that should be considered in a real trading situation, such as transaction costs, slippage, risk management, and overfitting.


import ccxt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fetch historical data from the exchange
def fetch_historical_data(exchange, symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return data

# Calculate technical indicators (e.g., MACD) and add them as features
def add_technical_indicators(dataframe):
    # Example: Adding MACD as a feature
    short_window = 12
    long_window = 26
    signal_window = 9
    ewma_short = dataframe['close'].ewm(span=short_window, adjust=False).mean()
    ewma_long = dataframe['close'].ewm(span=long_window, adjust=False).mean()
    macd_line = ewma_short - ewma_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    dataframe['macd'] = macd_line
    dataframe['signal'] = signal_line
    dataframe['macd_hist'] = macd_line - signal_line
    # Add more indicators as needed
    return dataframe

# Define a function to label our dataset with buy (1), sell (0), or hold (2)
def label_data(dataframe):
    dataframe['signal_shifted'] = dataframe['macd'].shift(-1) # Shift signal to label for next interval
    dataframe['target'] = np.where(dataframe['signal_shifted'] > dataframe['signal'], 1, 0)
    dataframe.dropna(inplace=True) # Drop rows with NaN values resulting from the shift
    return dataframe

# Load historical data
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'  # 1-hour intervals
data = fetch_historical_data(exchange, symbol, timeframe)
data_with_indicators = add_technical_indicators(data)
labeled_data = label_data(data_with_indicators)

# Prepare features and labels for training
features = labeled_data[['open', 'high', 'low', 'close', 'volume', 'macd', 'signal', 'macd_hist']]
labels = labeled_data['target']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.2f}')

# Now you would integrate this model with your trading bot logic to predict and execute trades
