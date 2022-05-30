from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from talos import Scan

top_words = 5000
(X_train, Y_train), (X_val, Y_val) = imdb.load_data(num_words=top_words)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)


def movie_review(x_train, y_train, x_val, y_val, params):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense

    from tensorflow.keras.layers import Embedding
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(params['first_neuron'], activation=params['activation']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # Fit the model
    out = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val), epochs=2,
                    batch_size=params['batch_size'], verbose=0)
    return out, model


p = {
    "first_neuron": [125, 250, 300],
    "activation": ["relu", "elu"],
    "batch_size": [32, 64, 128],
}

exp_name = "movie_review_test"


t = Scan(
    x=X_train,
    y=Y_train,
    x_val=X_val,
    y_val=Y_val,
    params=p,
    model=movie_review,
    experiment_name=exp_name,
   )
