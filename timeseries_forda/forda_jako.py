
import numpy as np
from jako import DistributedScan


def readucr(filename):
    data = np.loadtxt(filename, delimiter='\t')
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'

x_train, y_train = readucr(root_url + 'FordA_TRAIN.tsv')
x_val, y_val = readucr(root_url + 'FordA_TEST.tsv')


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_val[y_val == -1] = 0


def FordaModel(x_train, y_train, x_val, y_val, params):
    from tensorflow import keras
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=params['first_filter'],
                                kernel_size=3, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64,
                                kernel_size=3, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64,
                                kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap)

    model = keras.models.Model(inputs=input_layer,
                               outputs=output_layer)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    out = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=params['batch_size'],
        epochs=params['epochs'],

        verbose=2,
    )
    return out, model


p = {'first_filter': [64, 128, 256],
     'batch_size': [32, 64, 128],
     'epochs': [100, 500, 1000],
     'min_lr': [4e-6, 1e-5, 7e-5, 0.01]
     }

exp_name = 'forda_jako'

t = DistributedScan(
    x=x_train,
    y=y_train,
    x_val=x_val,
    y_val=y_val,
    params=p,
    model=FordaModel,
    experiment_name=exp_name,
    config='config.json')
