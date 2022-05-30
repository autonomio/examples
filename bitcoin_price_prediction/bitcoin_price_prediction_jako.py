import pandas as pd
from jako import DistributedScan


data = pd.read_csv('crypto_tradinds.csv')
btc_data = data[data['ticker'] == 'BTC']
btc_data_0 = btc_data[data['price_btc'] == 0]
drop_columns_list = btc_data.nunique()[btc_data.nunique() <= 2].index
btc_data.drop(drop_columns_list, axis=1, inplace=True)


def data_preproc_and_split(data, n):
    col = []

    for i in range(n):
        col.append('price' + str(i))
        col.append('volume' + str(i))

    train = pd.DataFrame(columns=col)
    target = pd.DataFrame(columns=['date', 'price'])
    pred_convert = pd.DataFrame(columns=['date', 'price'])

    # Preprocessing of data
    for i in range(1, len(data)-n-1):
        def_nom = data.loc[i-1, 'price_usd']

        for j in range(n):
            train.loc[i, 'price' + str(j)] = data.loc[i+j,
                                                      'price_usd']/def_nom-1
            vstr = 'volume' + str(j)
            train.loc[i, vstr] = data.loc[i+j, 'volume']/data.loc[i+j,
                                                                  'market_cap']

        target.loc[i, 'price'] = data.loc[i+n+1, 'price_usd']/def_nom-1
        target.loc[i, 'date'] = data.loc[i+n+1, 'trade_date']
        # Save start prices for convertation prediction resalt to valid prices
        pred_convert.loc[i, 'price'] = def_nom
        pred_convert.loc[i, 'date'] = data.loc[i+n+1, 'trade_date']

    # Data split
    x_train = train.iloc[:train.shape[0]-100]
    x_valid = train.iloc[train.shape[0]-100:]
    y_train = target.iloc[:target.shape[0]-100]
    y_valid = target.iloc[target.shape[0]-100:]
    y_train.drop(['date'], axis=1, inplace=True)
    y_valid.drop(['date'], axis=1, inplace=True)

    # Convert shape of data for LSTM model
    x_train = x_train.to_numpy().reshape((x_train.shape[0], n, 2))
    x_valid = x_valid.to_numpy().reshape((x_valid.shape[0], n, 2))
    return x_train, x_valid, y_train, y_valid, target, pred_convert


def bitcoin_model(x_train, y_train, x_val, y_val, params):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    mod = Sequential()
    mod.add(LSTM(params['first_neuron'],
                 return_sequences=True, input_shape=(25, 2)))
    mod.add(LSTM(64))
    mod.add(Dropout(params['dropout']))
    mod.add(Dense(128, activation='relu'))
    mod.add(Dense(1))
    mod.compile(optimizer='adam', loss='mse')
    mod.fit(x_train, y_train,
            batch_size=params["batch_size"],
            validation_data=(x_val, y_val),
            epochs=75, shuffle=False, verbose=2)

    return mod


n = 25  # chunk from the dataset used to train the model
x, x_val, y, y_val, target, pred_convert = data_preproc_and_split(btc_data, n)
p = {
    "first_neuron": [16, 32, 48],
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [128, 256, 512],
    'epochs': [25, 50, 75, 100]
}

exp_name = 'bitcoin_price_prediction'

t = DistributedScan(
    x=x,
    y=y,
    x_val=x_val,
    y_val=y_val,
    params=p,
    model=bitcoin_model,
    experiment_name=exp_name,
    config='config.json'
)
