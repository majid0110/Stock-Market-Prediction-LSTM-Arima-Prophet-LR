import numpy as np
import pandas as pd
import fastai
import re
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from pmdarima.arima import auto_arima
from prophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from prettytable import PrettyTable

df = pd.read_csv('G.Barekat.Pharm.csv')
df.head()

query = pd.read_csv('test.csv', nrows= 1)
query.head()

df.shape

df.info()

df.any().isnull()

df['<PER>'].value_counts()

sorted(df['<PER>'].unique())

df.drop(['<TICKER>','<PER>'], axis= 1, inplace= True)
df['<VALUE>'] = df['<VALUE>'].astype('float64')
df['<DATE>']= pd.to_datetime(df['<DTYYYYMMDD>'], format='%Y%m%d', errors='coerce')
df.drop(['<DTYYYYMMDD>'], axis= 1, inplace= True)
df.head()

query.drop(['<TICKER>','<PER>'], axis= 1, inplace= True)
query['<VALUE>'] = query['<VALUE>'].astype('float64')
query['<DATE>']= pd.to_datetime(query['<DTYYYYMMDD>'], format='%Y%m%d', errors='coerce')
query.drop(['<DTYYYYMMDD>'], axis= 1, inplace= True)
query.head()

df['<PROFIT>'] = df['<CLOSE>'] - df['<OPEN>']
df['label'] = 0

for i in range(0,len(df)):
  if (df['<PROFIT>'][i]>= 0):
    df['label'][i] = 1

df.head(10)

query['<PROFIT>'] = query['<CLOSE>'] - query['<OPEN>']
query['label'] = 0

if (query['<PROFIT>'][0]>= 0):
    query['label'][0] = 1

query.head()

df['label'].value_counts()

rcParams['figure.figsize'] = 20,10

#setting index as date
df.index = df['<DATE>']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['<CLOSE>'], label='Close Price history')

#sorting
df= df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['<DATE>','<CLOSE>','<OPEN>','label'])

for i in range(0,len(df)):
     new_data['<DATE>'][i] = df['<DATE>'][i]
     new_data['<OPEN>'][i] = df['<OPEN>'][i]
     new_data['<CLOSE>'][i] = df['<CLOSE>'][i]
     new_data['label'][i] = df['label'][i]

new_data

#creating a separate dataset
copy_query = pd.DataFrame(index=range(0,len(query)),columns=['<DATE>','<CLOSE>','<OPEN>','label'])

copy_query['<DATE>'][0] = query['<DATE>'][0]
copy_query['<OPEN>'][0] = query['<OPEN>'][0]
copy_query['<CLOSE>'][0] = query['<CLOSE>'][0]
copy_query['label'][0] = query['label'][0]

copy_query

def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        if n == 'Week':
            df[targ_pre + n] = fld.dt.isocalendar().week
        else:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


add_datepart(new_data, '<DATE>')
new_data.columns = ['Close', 'Open', 'Label', 'Year', 'Month', 'Week', 'Day', 'Dayofweek',
                    'Dayofyear', 'Is_month_end', 'Is_month_start',
                    'Is_quarter_end', 'Is_quarter_start', 'Is_year_end',
                    'Is_year_start', 'Elapsed']

new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

new_data

#create features
add_datepart(copy_query, '<DATE>')
copy_query.columns = ['Close', 'Open', 'Label' , 'Year', 'Month', 'Week', 'Day', 'Dayofweek',
                    'Dayofyear', 'Is_month_end', 'Is_month_start',
                    'Is_quarter_end', 'Is_quarter_start', 'Is_year_end',
                    'Is_year_start', 'Elapsed']

copy_query.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

copy_query

#split into train and validation
train = new_data[:663]
valid = new_data[663:]

train

valid

y_train = train['Close']
y_valid = valid['Close']

x_train = train.drop(['Label','Open','Close'], axis= 1)
x_valid = valid.drop(['Label','Open', 'Close'], axis= 1)

y_query = copy_query['Close']
x_query = copy_query.drop(['Label','Open','Close'], axis= 1)

model = LinearRegression()
model.fit(x_train,y_train)

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

train.index = new_data[:663].index
valid.index = new_data[663:].index

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])

valid['Predicted Profit'] = valid['Predictions'] - valid['Open']
valid['Predicted Labels'] = 0

for i in range(663,921):
  if (valid['Predicted Profit'][i]>= 0):
    valid['Predicted Labels'][i] = 1

valid

true_labels = valid['Label'].values.tolist()
pred_labels = valid['Predicted Labels'].values.tolist()

print(confusion_matrix(true_labels, pred_labels))

acc = accuracy_score(true_labels, pred_labels)
print(acc)

copy_query['Predictions'] = model.predict(x_query)
copy_query['Predicted Profit'] = copy_query['Predictions'] - copy_query['Open']
copy_query['Predicted Labels'] = 0

if (copy_query['Predicted Profit'][0]>= 0):
  copy_query['Predicted Labels'][0] = 1

copy_query

y_query_pred = copy_query['Predicted Labels'].values.tolist()
y_query = y_query.values.tolist()
acc = accuracy_score(y_query, y_query_pred)
print(acc)

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

x_train[x_train.columns]= scaler.fit_transform(x_train[x_train.columns])
x_valid[x_valid.columns]= scaler.fit_transform(x_valid[x_valid.columns])

x_train

x_query[x_query.columns]= scaler.fit_transform(x_query[x_query.columns])
x_query

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])

valid['Predicted Profit'] = valid['Predictions'] - valid['Open']
valid['Predicted Labels'] = 0

for i in range(663,921):
  if (valid['Predicted Profit'][i]>= 0):
    valid['Predicted Labels'][i] = 1

valid

true_labels = valid['Label'].values.tolist()
pred_labels = valid['Predicted Labels'].values.tolist()

print(confusion_matrix(true_labels, pred_labels))

acc = accuracy_score(true_labels, pred_labels)
acc

copy_query['Predictions'] = model.predict(x_query)
copy_query['Predicted Profit'] = copy_query['Predictions'] - copy_query['Open']
copy_query['Predicted Labels'] = 0

if (copy_query['Predicted Profit'][0]>= 0):
  copy_query['Predicted Labels'][0] = 1

copy_query

y_query_pred = copy_query['Predicted Labels'].values.tolist()
acc = accuracy_score(y_query, y_query_pred)
print(acc)

df.columns = ['First', 'High', 'Low' , 'Close', 'Value', 'Vol', 'Openint',
              'Open', 'Last', 'Date', 'Profit', 'Label']

query.columns = ['First', 'High', 'Low' , 'Close', 'Value', 'Vol', 'Openint',
              'Open', 'Last', 'Date', 'Profit', 'Label']

#split into train and validation
training = (df[:663])['Close']
validation = (df[663:])['Close']

validation

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12
                   ,start_P=0, seasonal=True,d=1, D=1,
                   trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)
forecast = model.predict(n_periods=259)
forecast = pd.DataFrame(forecast,index = range(663, 922),columns=['Prediction'])

copy_query['Predictions'] = forecast.tail(1).values
forecast.drop(forecast.tail(1).index,inplace=True)

rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))
rms

#plot
plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(forecast['Prediction'])

valid['Predictions'] = forecast['Prediction']
valid['Predicted Profit'] = valid['Predictions'] - valid['Open']
valid['Predicted Labels'] = 0

for i in range(663,921):
  if (valid['Predicted Profit'][i]>= 0):
    valid['Predicted Labels'][i] = 1

valid

true_labels = valid['Label'].values.tolist()
pred_labels = valid['Predicted Labels'].values.tolist()

print(confusion_matrix(true_labels, pred_labels))

acc = accuracy_score(true_labels, pred_labels)
acc

copy_query['Predicted Profit'] = copy_query['Predictions'] - copy_query['Open']
copy_query['Predicted Labels'] = 0

if (copy_query['Predicted Profit'][0]>= 0):
  copy_query['Predicted Labels'][0] = 1

copy_query

y_query_pred = copy_query['Predicted Labels'].values.tolist()
acc = accuracy_score(y_query, y_query_pred)
print(acc)

#creating dataframe
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(df)):
    new_data['Date'][i] = df['Date'][i]
    new_data['Close'][i] = df['Close'][i]

new_data.index = new_data['Date']

#preparing data
new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

new_data

#creating dataframe
copy_query = pd.DataFrame(index=range(0,len(query)),columns=['Date', 'Close'])

copy_query['Date'][0] = query['Date'][0]
copy_query['Close'][0] = query['Close'][0]

copy_query.index = copy_query['Date']

#preparing data
copy_query.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

copy_query

#train and validation
training = new_data[:663]
validation = new_data[663:]

x_query = copy_query
x_query

#fit the model
model = Prophet()
model.fit(training)

#predictions
close_prices = model.make_future_dataframe(periods=len(validation)+1)
forecast = model.predict(close_prices)

#rmse
forecast_valid = forecast['yhat'][663:921]
copy_query['Predictions'] = forecast['yhat'][921:].values
rms=np.sqrt(np.mean(np.power((np.array(validation['y'])-np.array(forecast_valid)),2)))
rms

#plot
validation['Predictions'] = 0
valid['Predictions'] = forecast_valid.values
validation['Predictions'] = forecast_valid.values

plt.plot(training['y'])
plt.plot(validation[['y', 'Predictions']])

valid['Predicted Profit'] = valid['Predictions'] - valid['Open']
valid['Predicted Labels'] = 0

for i in range(663,921):
  if (valid['Predicted Profit'][i]>= 0):
    valid['Predicted Labels'][i] = 1

valid

true_labels = valid['Label'].values.tolist()
pred_labels = valid['Predicted Labels'].values.tolist()

print(confusion_matrix(true_labels, pred_labels))

acc = accuracy_score(true_labels, pred_labels)
acc

copy_query

copy_query['Predicted Profit'] = copy_query['Predictions'] - query['Open'].values
copy_query['Predicted Labels'] = 0

if (copy_query['Predicted Profit'][0]>= 0):
  copy_query['Predicted Labels'][0] = 1

copy_query

y_query_pred = copy_query['Predicted Labels'].values.tolist()
acc = accuracy_score(y_query, y_query_pred)
print(acc)

new_data.rename(columns={'y': 'Close', 'ds': 'Date'}, inplace=True)
new_data.drop('Date', axis=1, inplace=True)

copy_query.rename(columns={'y': 'Close', 'ds': 'Date'}, inplace=True)
copy_query.drop(['Date', 'Predictions', 'Predicted Profit', 'Predicted Labels'], axis=1, inplace=True)

new_data = pd.concat([new_data, copy_query], ignore_index=True)

#creating train and test sets
dataset = new_data.values

training = dataset[0:663,:]
validation = dataset[663:,:]

training.shape

validation.shape

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(training)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_train.shape

y_train.shape

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 258 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(validation) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

closing_price.shape

training = new_data[:663]
validation = new_data[663:921]
copy_query['Predictions'] = closing_price[-1]

rms=np.sqrt(np.mean(np.power(((validation-closing_price[:-1]).values),2)))
rms

#for plotting

validation['Predictions'] = closing_price[:-1]
valid['Predictions'] = closing_price[:-1]
plt.plot(training['Close'])
plt.plot(validation[['Close','Predictions']])

valid['Predicted Profit'] = valid['Predictions'] - valid['Open']
valid['Predicted Labels'] = 0

for i in range(663,921):
  if (valid['Predicted Profit'][i]>= 0):
    valid['Predicted Labels'][i] = 1

valid

true_labels = valid['Label'].values.tolist()
pred_labels = valid['Predicted Labels'].values.tolist()

print(confusion_matrix(true_labels, pred_labels))

acc = accuracy_score(true_labels, pred_labels)
acc

copy_query['Predicted Profit'] = copy_query['Predictions'] - query['Open'].values
copy_query['Predicted Labels'] = 0

if (copy_query['Predicted Profit'][0]>= 0):
  copy_query['Predicted Labels'][0] = 1

copy_query

y_query_pred = copy_query['Predicted Labels'].values.tolist()
acc = accuracy_score(y_query, y_query_pred)
print(acc)
