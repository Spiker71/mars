import logging
import sys
from datetime import datetime, timedelta
import numpy as np
import talib
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException
from scipy.signal import argrelextrema
import argparse

# Установка кодировки для вывода в консоль
sys.stdout.reconfigure(encoding='utf-8')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)

def get_historical_klines(client, symbol, interval, start_str, end_str=None):
    """Получение исторических данных"""
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        return klines
    except BinanceAPIException as e:
        logging.error(f"Ошибка при получении исторических данных: {e}")
        return []

def calculate_fibonacci_levels(data):
    """Расчет уровней Фибоначчи"""
    max_price = max(data)
    min_price = min(data)
    diff = max_price - min_price
    levels = {
        '0.0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50.0%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100.0%': min_price
    }
    return levels

def find_trade_signals(data, levels, symbol):
    """Поиск точек входа на основе уровней Фибоначчи"""
    signals = []
    for i in range(1, len(data)):
        if data[i-1] > levels['38.2%'] and data[i] <= levels['38.2%']:
            signals.append(('Buy', i, symbol, levels['38.2%'], levels['23.6%'], levels['61.8%']))
        elif data[i-1] < levels['61.8%'] and data[i] >= levels['61.8%']:
            signals.append(('Sell', i, symbol, levels['61.8%'], levels['50.0%'], levels['100.0%']))
    return signals

def wave_analysis(data):
    """Простой волновой анализ на основе локальных экстремумов"""
    max_idx = argrelextrema(np.array(data), np.greater, order=5)[0]
    min_idx = argrelextrema(np.array(data), np.less, order=5)[0]
    waves = []
    for i in range(len(max_idx)-1):
        waves.append(('Wave', max_idx[i], min_idx[i], data[max_idx[i]], data[min_idx[i]]))
    return waves

def calculate_indicators(data, volume):
    """Расчет индикаторов MACD, RSI, SMA, EMA и объема"""
    close_prices = np.array(data)
    macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = talib.RSI(close_prices, timeperiod=14)
    sma = talib.SMA(close_prices, timeperiod=50)
    ema = talib.EMA(close_prices, timeperiod=50)
    vol_sma = talib.SMA(np.array(volume), timeperiod=20)
    return macd, macd_signal, macd_hist, rsi, sma, ema, vol_sma

def create_chart(data, levels, symbol, action, index, entry, take_profit, stop_loss, macd, macd_signal, rsi, sma, ema, volume, vol_sma):
    """Создание графика с уровнями Фибоначчи и точками входа/выхода"""
    plt.figure(figsize=(12, 9))
    
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Price')
    plt.plot(sma, label='SMA')
    plt.plot(ema, label='EMA')
    plt.axhline(y=entry, color='b', linestyle='--', label='Entry')
    plt.axhline(y=take_profit, color='g', linestyle='--', label='Take Profit')
    plt.axhline(y=stop_loss, color='r', linestyle='--', label='Stop Loss')
    for level, value in levels.items():
        plt.axhline(y=value, linestyle=':', label=f'Fibonacci {level}')
    plt.legend()
    plt.title(f'{symbol} - {action} Signal at Index {index}')

    plt.subplot(4, 1, 2)
    plt.plot(macd, label='MACD')
    plt.plot(macd_signal, label='MACD Signal')
    plt.legend()
    plt.title('MACD')

    plt.subplot(4, 1, 3)
    plt.plot(rsi, label='RSI')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.legend()
    plt.title('RSI')

    plt.subplot(4, 1, 4)
    plt.plot(volume, label='Volume')
    plt.plot(vol_sma, label='Volume SMA')
    plt.legend()
    plt.title('Volume')

    plt.tight_layout()
    plt.savefig(f'{symbol}_{action}_{index}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Binance trading bot with Fibonacci and indicators')
    parser.add_argument('--api_key', type=str, required=True, help='Your Binance API key')
    parser.add_argument('--api_secret', type=str, required=True, help='Your Binance API secret')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], help='List of symbols to analyze')
    parser.add_argument('--intervals', type=str, nargs='+', default=[
        Client.KLINE_INTERVAL_1MINUTE,
        Client.KLINE_INTERVAL_5MINUTE,
        Client.KLINE_INTERVAL_15MINUTE,
        Client.KLINE_INTERVAL_30MINUTE,
        Client.KLINE_INTERVAL_1HOUR,
        Client.KLINE_INTERVAL_2HOUR,
        Client.KLINE_INTERVAL_4HOUR,
        Client.KLINE_INTERVAL_6HOUR,
        Client.KLINE_INTERVAL_8HOUR,
        Client.KLINE_INTERVAL_12HOUR,
        Client.KLINE_INTERVAL_1DAY,
        Client.KLINE_INTERVAL_3DAY,
        Client.KLINE_INTERVAL_1WEEK,
        Client.KLINE_INTERVAL_1MONTH
    ], help='List of intervals to analyze')
    parser.add_argument('--start_time', type=str, default=(datetime.now() - timedelta(days=30)).strftime("%d %b, %Y %H:%M:%S"), help='Start time for historical data')
    parser.add_argument('--end_time', type=str, default=datetime.now().strftime("%d %b, %Y %H:%M:%S"), help='End time for historical data')
    
    args = parser.parse_args()

    # Создание клиента Binance
    client = Client(args.czs8NPf9uo1va2Sg4HB5NCWFO7XGNtP8RPHWLWU8eWqNw0XhqjCsPhJreJfaEMhv, args.v0Onk3jFT4G5Q4vufMt3eDqT2r2cKKW4NoOQC53uLNSfjRcBHfqdmYBrHaFa3Udx)

    for symbol in args.symbols:
        for interval in args.intervals:
            # Получение исторических данных
            klines = get_historical_klines(client, symbol, interval, args.start_time, args.end_time)
            if not klines:
                logging.error(f"Не удалось получить исторические данные для {symbol}. Прерывание работы.")
                continue

            close_prices = np.array([float(kline[4]) for kline in klines])
            volume = np.array([float(kline[5]) for kline in klines])

            # Рассчет уровней Фибоначчи
            fibonacci_levels = calculate_fibonacci_levels(close_prices)

            # Поиск точек входа
            signals = find_trade_signals(close_prices, fibonacci_levels, symbol)

            # Волновой анализ
            waves = wave_analysis(close_prices)

            # Рассчет индикаторов
            macd, macd_signal, macd_hist, rsi, sma, ema, vol_sma = calculate_indicators(close_prices, volume)

            # Вывод сигналов и создание графиков
            for signal in signals:
                action, index, symbol, entry, take_profit, stop_loss = signal
                logging.info(f"Signal: {action} for {symbol} at index {index} (entry: {entry}, take profit: {take_profit}, stop loss: {stop_loss})")
                create_chart(close_prices, fibonacci_levels, symbol, action, index, entry, take_profit, stop_loss, macd, macd_signal, rsi, sma, ema, volume, vol_sma)

if __name__ == '__main__':
    main()
