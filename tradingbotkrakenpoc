import krakenex
from pykrakenapi import KrakenAPI
import numpy as np
import pandas as pd
import time

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    ewma_short = data['close'].ewm(span=short_window).mean()
    ewma_long = data['close'].ewm(span=long_window).mean()
    macd_line = ewma_short - ewma_long
    signal_line = macd_line.ewm(span=signal_window).mean()
    return macd_line, signal_line

# Initialize the Kraken API object
api_key = 'your_api_key'        # Your Kraken API key
api_secret = 'your_api_secret'  # Your Kraken API secret

kraken = krakenex.API(api_key=api_key, api_secret=api_secret)
k = KrakenAPI(kraken)

symbol = 'XXRPXXBT'
usd_symbol = 'XXBTZUSD'
timeframe = '1m'
previous_trade = 'sell'
current_timeframe = 1

# Fetch the BTC price in USD
btc_price_usd = float(k.get_ticker_information(usd_symbol)['c'][0][0])

# Initialize balance
xrp_usd_balance = 45
btc_usd_balance = 45

xrp_balance = xrp_usd_balance / btc_price_usd
btc_balance = btc_usd_balance / btc_price_usd

last_xrp_price = 0

while xrp_balance > 0 or btc_balance > 0:
    # Fetch historical data and calculate MACD
    ohlcv, last = k.get_ohlc_data(symbol, interval=1, ascending=True)
    data = ohlcv.tail(100)  # Get the most recent 100 minutes
    macd_line, signal_line = calculate_macd(data)

    current_xrp_price = float(data['close'].iloc[-1])

    # Check for a buy or sell signal
    if macd_line.iloc[-1] > signal_line.iloc[-1] and previous_trade == 'sell' and btc_balance > 0 and current_xrp_price > last_xrp_price:
        # Buy signal
        print("Buy signal")
        
        # Execute buy order
        order = k.add_standard_order(symbol, type='buy', ordertype='limit', price=str(current_xrp_price), volume=str(btc_balance / current_xrp_price))
        print(order)

        # Update balances
        xrp_balance += btc_balance / current_xrp_price
        btc_balance = 0
        previous_trade = 'buy'
        last_xrp_price = current_xrp_price

    elif macd_line.iloc[-1] < signal_line.iloc[-1] and previous_trade == 'buy' and xrp_balance > 0 and current_xrp_price < last_xrp_price:
        # Sell signal
        print("Sell signal")

        # Execute sell order
        order = k.add_standard_order(symbol, type='sell', ordertype='limit', price=str(current_xrp_price), volume=str(xrp_balance))
        print(order)

        # Update balances
        btc_balance += xrp_balance * current_xrp_price
        xrp_balance = 0
        previous_trade = 'sell'
        last_xrp_price = current_xrp_price

    # Sleep for 1 minute
    time.sleep(60)
