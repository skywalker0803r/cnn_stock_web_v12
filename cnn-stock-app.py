from flask import Flask, render_template
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas_datareader.data as web
from finta import TA
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from keras import backend as K
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go

# === 初始化 Flask App ===
app = Flask(__name__, static_url_path='/static')

# === 定義常數與路徑 ===
MODEL_PATH = './CNN_stock_model/model.h5'
SCALER_PATH = './CNN_stock_model/scaler.pkl'
SID = '^TWII'
DAYS = 15
FEATURES = [
    'RSI', 'Williams %R', 'SMA', 'EMA', 'WMA', 'HMA',
    'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CFI', 'DMI', 'SAR'
]

# === 資料處理 ===
def load_stock_data():
    one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = yf.download(SID, start=one_year_ago)
    ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
    df['RSI'] = TA.RSI(ohlcv)
    df['Williams %R'] = TA.WILLIAMS(ohlcv)
    df['SMA'] = TA.SMA(ohlcv)
    df['EMA'] = TA.EMA(ohlcv)
    df['WMA'] = TA.WMA(ohlcv)
    df['HMA'] = TA.HMA(ohlcv)
    df['TEMA'] = TA.TEMA(ohlcv)
    df['CCI'] = TA.CCI(ohlcv)
    df['CMO'] = TA.CMO(ohlcv)
    df['MACD'] = TA.MACD(ohlcv)['MACD'] - TA.MACD(ohlcv)['SIGNAL']
    df['PPO'] = TA.PPO(ohlcv)['PPO'] - TA.PPO(ohlcv)['SIGNAL']
    df['ROC'] = TA.ROC(ohlcv)
    df['CFI'] = TA.CFI(ohlcv)
    df['DMI'] = TA.DMI(ohlcv)['DI+'] - TA.DMI(ohlcv)['DI-']
    df['SAR'] = TA.SAR(ohlcv)
    return df.dropna()

def prepare_features(df):
    scaler = joblib.load(SCALER_PATH)
    df_scaled = pd.DataFrame(scaler.transform(df[FEATURES]), columns=FEATURES, index=df.index)
    Xs = np.array([df_scaled.iloc[i:i+DAYS].values for i in range(len(df_scaled) - DAYS + 1)])
    return Xs.reshape(-1, DAYS, len(FEATURES), 1)

# === 預測與圖形繪製 ===
def predict_signals(Xs, close_prices):
    model = load_model(MODEL_PATH)
    preds = model.predict(Xs)
    signal = [np.argmax(p) for p in preds]
    result_df = pd.DataFrame(preds, columns=['HOLD', 'BUY', 'SELL'])
    result_df['Close'] = close_prices[-len(preds):].values
    result_df['SIGNAL'] = signal
    result_df.index = close_prices[-len(preds):].index
    return result_df

def plot_prediction(df):
    recent = df[-40:]
    buy = recent[recent['SIGNAL'] == 1]
    sell = recent[recent['SIGNAL'] == 2]

    fig = go.Figure()

    # 收盤價線：黑色線條
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='black')
    ))

    # 買進訊號：綠色正三角
    fig.add_trace(go.Scatter(
        x=buy.index,
        y=buy['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

    # 賣出訊號：紅色倒三角
    fig.add_trace(go.Scatter(
        x=sell.index,
        y=sell['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

    fig.update_layout(
        title="Buy/Sell Prediction (Interactive)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            bgcolor='white',
            bordercolor='lightgray',
            borderwidth=1
        )
    )

    fig.write_html("./static/predict_result.html", include_plotlyjs='cdn')

# === 主邏輯 ===
def create_result():
    df = load_stock_data()
    Xs = prepare_features(df)
    result_df = predict_signals(Xs, df['Close'])
    plot_prediction(result_df)
    return result_df

# === Flask route ===
@app.route('/')
def render_page():
    K.clear_session()
    result = create_result()
    K.clear_session()
    result['date'] = result.index.strftime('%Y-%m-%d')
    return render_template('cnn-stock-web.html', data=result.tail(40).values)

# === 執行 ===
if __name__ == '__main__':
    app.run(debug=False, port=int(os.getenv('PORT', 5015)))
# 我想要讓網頁上的股價走勢圖可以放大特定區間
