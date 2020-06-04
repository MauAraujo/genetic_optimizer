import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from numpy import std
from tqdm import tqdm


def get_dataset(dataset):
    # Función para abrir el dataset y limpiarlo,
    # transformando los valores categóricos a discretos

    df = pd.read_csv(dataset)
    df['class'] = pd.Categorical(df['class'])
    df['class'] = df['class'].cat.codes
    class_name = df.pop('class') # Separar el valor de clase como el target
    dataset = df
    return dataset.values, class_name.values


def compile_model(network):
    # Función para compilar el modelo

    model = keras.Sequential()

    # Agregar capas
    for _ in range(network['layers']):
        model.add(layers.Dense(network['neurons'], activation='sigmoid'))

    # Agregar un optimizador utilizando el lr y momentum
    opt = keras.optimizers.SGD(learning_rate=network['lr'],
                               momentum=network['momentum'])

    # Definir función de loss
    loss = tf.keras.losses.MeanSquaredError()

    # Compilar
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


def train_and_evaluate(network, model, X_train, y_train, X_test, y_test):
    # Función para entrenar y evaluar el modelo

    # Se entrena al modelo con el número de épocas definido con los datos de prueba
    model.fit(X_train, y_train, epochs=network['epochs'],
              validation_data=(X_test, y_test), verbose=1)

    # Se evalua con el conjunto de validación
    _, test_acc = model.evaluate(X_test, y_test, verbose=1)
    return model, test_acc


def train(network, dataset):
    # Función principal, se realiza la validación cruzada

    scores, members = [], []
    n_splits = 10 # Valor de K para la validación cruzada
    X, y = get_dataset(dataset)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    pbar = tqdm(total=n_splits) # Mostrar barra de progreso

    # Ciclo de validación cruzada
    for train_index, test_index in skf.split(X, y):
        model = None # Resetear el modelo en cada iteración
        model = compile_model(network) # Compilar modelo

        # Dividir el dataset en entrenamiento y validación
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Entrenar y evaluar
        model, test_acc = train_and_evaluate(network, model, X_train, y_train,
                                             X_test, y_test)

        # Guardar resultados
        scores.append(test_acc)
        members.append(model)
        pbar.update(1)
    pbar.close()

    print('Precisión estimada %.3f (%.3f)' % (mean(scores), std(scores)))

    # Se regresa el promedio de todas las validaciones
    return mean(scores)
