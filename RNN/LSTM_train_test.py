import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.models import Sequential
##from sklearn.metrics import mean_squared_error
import math
##import keras.backend as K
    
##def koubei_loss(y_true, y_pred):
##    return K.mean(K.abs(y_pred - y_true)/(y_pred + y_true), axis=-1)

class LSTM_RNN:
    def __init__(self, batch_size, hidden_layers, init='he_uniform'):
        self.batch_size = batch_size
        self.nn = Sequential()
        self.nn.add(LSTM(hidden_layers, input_dim=week, return_sequences=True))  # , init=init))
        self.nn.add(LSTM(hidden_layers, input_dim=week))
##        self.nn.add(Dropout(0.2))
        self.nn.add(Dense(pred_days))  # , init=init))
        self.nn.compile(loss='mae', optimizer='rmsprop')

    def train(self, X, Y, nb_epoch):
        print('training LSTM_RNN...')
        self.nn.fit(X, Y, nb_epoch=nb_epoch, batch_size=self.batch_size, verbose=2)

    # def evaluate(self, X, Y):
    #     return self.nn.evaluate(X, Y, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        return self.nn.predict(X)


def load_dataset(path, cols):
    print('load %s...' % path)
    dataframe = pd.read_csv(path, usecols=cols, engine='c', header=None)
    return dataframe.values.astype('float32')


def split_data(ds, look_back, final_output):
    fea = ds[:, 0:look_back]
    pred = ds[:, look_back:look_back + final_output]
    return np.array(fea), np.array(pred)


np.random.seed(0)
week = 7
time_steps = 2
fea_days = time_steps * week
pred_days = week

batch_size = 64
hidden_layers = 7
nb_epoch = 150

scalar = load_dataset('tmp/scalar.csv', None)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(scalar)

###################################################
trds = load_dataset('tmp/trds_3w.csv', range(1, week * 3 + 1))
trds = scaler.transform(trds)

fea_tr, real_tr = split_data(trds, fea_days, pred_days)
# reshape input to be [samples, time steps, features]
fea_tr = np.reshape(fea_tr, (fea_tr.shape[0], time_steps, week))
print('training for batch_size={} and hidden_layers={}...'.format(batch_size, hidden_layers))
model = LSTM_RNN(batch_size, hidden_layers)
model.train(fea_tr, real_tr, nb_epoch)
pred_tr = model.predict(fea_tr)
pred_tr = scaler.inverse_transform(pred_tr)
real_tr = scaler.inverse_transform(real_tr)
##mse = mean_squared_error(pred_tr, real_tr)
##print('training MSE is %.2f.' % mse)
diff_pred = abs(pred_tr - real_tr)
plus_pred = pred_tr + pred_tr
L = sum(sum(diff_pred / plus_pred)) / (pred_tr.shape[0] * pred_tr.shape[1])
print('training LOSS is %.6f.' % L)

###################################################
tsds = load_dataset('tmp/tsds_3w.csv', range(1, fea_days + 1))
tsds = scaler.transform(tsds)

fea_ts = tsds
fea_ts = np.reshape(fea_ts, (fea_ts.shape[0], time_steps, week))
pred_ts = model.predict(fea_ts)
##pred_ts = scaler.inverse_transform(pred_ts)

fea_ts2 = np.append(tsds, pred_ts, axis=1)
fea_ts2 = fea_ts2[:,week:]
fea_ts2 = np.reshape(fea_ts2, (fea_ts2.shape[0], time_steps, week))
pred_ts2 = model.predict(fea_ts2)
pred_ts2 = scaler.inverse_transform(pred_ts2)

pred_ts = scaler.inverse_transform(pred_ts)
pred_tsx = np.append(pred_ts, pred_ts2, axis=1)
with open('upload/predict_mae_stack.csv', 'w') as wout:
    for i in range(2000):
        wout.write(str(i+1))
        for j in range(14):
            n = math.ceil(pred_tsx[i,j])
            if n < 0:
                n = 0
            wout.write(','+str(n))
        wout.write('\n')

        

# trainPredictPlot = np.empty_like(ds)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[fea_days:len(pred_m_tr) + fea_days, :] = pred_m_tr
#
# testPredictPlot = np.empty_like(ds)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[fea_days:len(pred_m_ts) + fea_days, :] = pred_m_ts
