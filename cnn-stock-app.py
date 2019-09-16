from flask import Flask,render_template
import os
import numpy as np
import io
from keras.models import Model
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import pandas_datareader.data as web
from finta import TA
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from keras import backend as K

#定義函數create_result()====================================
def create_result():
	#定義路徑
	model_path = './CNN_stock_model/Ricky_net_v12.h5'
	scaler_path = './CNN_stock_model/MinMax_scaler_v12.pkl'
	SID = '^TWII'
	#製作agent
	model = load_model(model_path)
	class Agent:
		def __init__(self,model):
			self.model = model
		def choose_action(self,features):
			predict = self.model.predict(features)
			return predict
	Ricky = Agent(model)
	#製作features
	df = web.DataReader(SID,'yahoo',start="2018-01-01")
	df = df.iloc[:,:5]
	ohlcv = df[['Open','High','Low','Close','Volume']]
	ohlcv.columns = ['open','high','low','close','volume']
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
	df = df.dropna(axis=0)
	features = df.columns[-15:].tolist()
	df = df[features]
	df['triple_barrier_signal'] = 0
	#features scaling
	min_max_scaler = joblib.load(scaler_path)
	df_minmax = min_max_scaler.transform(df)
	df_minmax = pd.DataFrame(df_minmax,index = df.index,columns = df.columns)
	df = df_minmax.drop('triple_barrier_signal',axis=1)
	#Xs prepare
	days = 15
	b_index = 0
	f_index = len(df)-days
	Xs = []
	for i in range(b_index ,f_index+1 ,1):
	  X = df.iloc[i:i+days,:][features]
	  X = np.array(X)
	  Xs.append(X)
	Xs = np.array(Xs)
	#Reshape X
	Xs = Xs.reshape(-1,days,len(features),1)
	#model predict
	predict = Ricky.choose_action(Xs)
	SIGNAL = [ np.argmax(i) for i in predict]
	#predict result
	df = df.iloc[-len(SIGNAL):]
	df['HOLD'] = predict[:,0]
	df['BUY'] = predict[:,1]
	df['SELL'] = predict[:,2]
	df = df.iloc[:,-3:]
	df['Close'] = web.DataReader(SID,'yahoo',start="2018-01-01")['Close'][-len(SIGNAL):].values
	df['SIGNAL'] = SIGNAL
	#繪圖
	t = df[-40:].copy()
	print(t)
	buy = t[t['SIGNAL']==1]['Close']
	sell = t[t['SIGNAL']==2]['Close']
	t['Close'].plot()
	plt.scatter(list(buy.index),list(buy.values),color='red',marker='^')
	plt.scatter(list(sell.index),list(sell.values),color='black')
	plt.savefig("./static/predict_result.png")
	return t
#=======================================================================

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def render_page():
	K.clear_session()
	table = create_result()
	K.clear_session()
	table['date'] = table.index
	data = table.values
	print(data.shape)
	return render_template('cnn-stock-web.html',data=data)

if __name__ == '__main__':
	app.run(debug=False,port=os.getenv('PORT',5015))