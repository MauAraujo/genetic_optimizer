import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from numpy import std
from tqdm import tqdm


def get_dataset(dataset):
    df = pd.read_csv(dataset)
    df['class'] = pd.Categorical(df['class'])
    df['class'] = df['class'].cat.codes
    class_name = df.pop('class')
    dataset = df
    return dataset.values, class_name.values


def compile_model(network):
    model = keras.Sequential()

    for _ in range(network['layers']):
        model.add(layers.Dense(network['neurons'], activation='sigmoid'))

    opt = keras.optimizers.SGD(learning_rate=network['lr'],
                               momentum=network['momentum'])
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


def train_and_evaluate(network, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=network['epochs'],
              validation_data=(X_test, y_test), verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return model, test_acc


def train(network, dataset):
    scores, members = [], []
    n_splits = 2
    X, y = get_dataset(dataset)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    pbar = tqdm(total=n_splits)
    for train_index, test_index in skf.split(X, y):
        model = None
        model = compile_model(network)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model, test_acc = train_and_evaluate(network, model, X_train, y_train,
                                             X_test, y_test)
        scores.append(test_acc)
        members.append(model)
        pbar.update(1)
    pbar.close()

    print('Estimated Accuracy %.3f (%.3f)' % (mean(scores), std(scores)))
    return mean(scores)
