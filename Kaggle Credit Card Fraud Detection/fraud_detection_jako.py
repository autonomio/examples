import numpy as np
from jako import DistributedScan
# Get the real data from https://www.kaggle.com/mlg-ulb/creditcardfraud/
fname = "creditcard.csv"

all_features = []
all_targets = []
with open(fname) as f:
    for i, line in enumerate(f):
        if i == 0:
            print("HEADER:", line.strip())
            continue  # Skip header
        fields = line.strip().split(",")
        all_features.append([float(v.replace('"', "")) for v in fields[:-1]])
        all_targets.append([int(fields[-1].replace('"', ""))])
        if i == 1:
            print("EXAMPLE FEATURES:", all_features[-1])

features = np.array(all_features, dtype="float32")
targets = np.array(all_targets, dtype="uint8")

num_val_samples = int(len(features) * 0.2)
x_train = features[:-num_val_samples]
y_train = targets[:-num_val_samples]
x_val = features[-num_val_samples:]
y_val = targets[-num_val_samples:]

counts = np.bincount(y_train[:, 0])

print(
    "Number of positive samples in training data: {} ({:.2f}% of total)".format(
        counts[1], 100 * float(counts[1]) / len(y_train)
    )
)

weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
mean = np.mean(x_train, axis=0)
x_train -= mean
x_val -= mean
std = np.std(x_train, axis=0)
x_train /= std
x_val /= std


def FraudDetection(x_train, y_train, x_val, y_val, params):
    from tensorflow import keras
    import numpy as np

    counts = np.bincount(y_train[:, 0])
    print(
        '''Number of positive samples in training data:
            {} ({:.2f}% of total)'''.format(
            counts[1], 100 * float(counts[1]) / len(y_train)
        )
    )

    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_val -= mean
    std = np.std(x_train, axis=0)
    x_train /= std
    x_val /= std
    model = keras.Sequential(
        [
            keras.layers.Dense(
                256, activation="relu", input_shape=(x_train.shape[-1],)
            ),
            keras.layers.Dense(params['first_neuron'], activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    metrics = [
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(params['lr']),
        loss="binary_crossentropy",
        metrics=metrics
    )

    class_weight = {0: weight_for_0, 1: weight_for_1}

    out = model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        verbose=2,
        validation_data=(x_val, y_val),
        class_weight=class_weight,
    )

    return out, model


p = {'first_neuron': [64, 128, 256],
     'batch_size': [1024, 2048, 4096],
     'epochs': [10, 30, 50],
     'lr': [4e-6, 1e-5, 7e-5, 0.01]
     }

exp_name = 'FraudDetection'

t = DistributedScan(
    x=x_train,
    y=y_train,
    x_val=x_val,
    y_val=y_val,
    params=p,
    model=FraudDetection,
    experiment_name=exp_name,
    config='config.json')
