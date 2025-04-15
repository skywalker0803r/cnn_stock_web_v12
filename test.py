from keras.models import load_model
import joblib

model = load_model('./CNN_stock_model/model.h5')
scaler = joblib.load('./CNN_stock_model/scaler.pkl')
print(model.summary())
print(scaler)
#ValueError: Kernel shape must have the same length as input, 
# but received kernel of shape (3, 3, 1, 32) 
# and input of shape (None, None, 15, 15, 1).
