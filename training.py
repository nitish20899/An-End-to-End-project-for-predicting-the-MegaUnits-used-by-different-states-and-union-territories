# final model for states other than AP and TELENGANA without mimax
# accuracy check

import pandas as pd
import time

def predict_rem(col_name):
    print('Creating a best model for predicting {} MU and Checking its accuracy with test data'.format(col_name))
    import pandas as pd
    import warnings

    warnings.filterwarnings("ignore")

    df3 = pd.read_csv('pycharm_data.csv')
    df3 = df3.drop(columns=['Unnamed: 0', 'date'])
    new_df = df3[col_name].copy()
    new_df

    # create a differenced series
    from pandas import Series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    differenced = difference(new_df, 1)
    differenced.head()

    X = []
    y = []
    for i in range(0, differenced.shape[0] - 48):
        X.append(differenced.iloc[i:i + 48])  # taking 48 rows for training incrementally
        y.append(differenced.iloc[i + 48])

    import numpy as np
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    y = np.reshape(y, (len(y), 1))

    X_train, X_test = X[:-480], X[-480:]
    y_train, y_test = y[:-480], y[-480:]

    for i in range(len(y_test)):
        y_test_diff = y_test[i] + new_df[1786 + i]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # reshaping the data to 3d
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    import warnings
    warnings.filterwarnings("ignore")
    import keras
    from keras.layers import LSTM, Activation, Dense
    from keras import optimizers
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.layers import ReLU, LeakyReLU
    import h5py
    from keras.models import load_model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM
    from kerastuner.tuners import BayesianOptimization

    def build_model(hp):
        for i in range(hp.Int('num_layers', 2, 5)):
            model = keras.Sequential()
            model.add(LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=300, step=32)))
            model.add(ReLU())
            model.add(Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.1,
                max_value=0.5,
                default=0.1,
                step=0.05)))
            model.add(Dense(1, activation='linear'))
            model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mean_absolute_error',
                metrics=['mse'])
        return model

    bayesian_opt_tuner = BayesianOptimization(
        build_model,
        objective='mse',
        max_trials=3,
        executions_per_trial=1,
        overwrite=True)

    bayesian_opt_tuner.search(X_train, y_train, epochs=1000, validation_data=(X_test, y_test),
                              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)], verbose=0,
                              batch_size=5)

    best_model = bayesian_opt_tuner.get_best_models(num_models=1)
    model = best_model[0]

    predictions = model.predict(X_test)
    final_pred = []
    for i in range(len(predictions)):
        y_test_diff = predictions[i] + new_df[1786 + i]
        final_pred.append(y_test_diff[0])
    final_pred

    y_test_orig = []
    for i in range(len(y_test)):
        y_test_diff = y_test[i] + new_df[1786 + i]
        y_test_orig.append(y_test_diff[0])
    y_test_orig

    total = 0
    for i, j in zip(y_test_orig, final_pred):
        value = abs(i - j) / abs(i)
        total += value
    error = float(total * 100 / (len(y_test_orig)))  # calculate mape
    mape = round(error, 1)  # round to 3 significant figures
    accuracy = 100 - mape  # Calculate accuracy

    import matplotlib.pyplot as plt
    print("The LSTM's accuracy in predicting the mega units on test data : " + str(accuracy) + "%")
    print(' ')
    df = pd.DataFrame(y_test_orig, final_pred).reset_index()
    df.columns = ['original', 'predicted']
    df.plot(title='Original data vs Predicted data (for test data)')
    plt.show()
    model.save('models_with_tuner\\' + col_name + '_model_75.h5')
    print(' ')
    print('================================================================================')

    print('')
    print('Now training the model with 100% data for future predictions....')
    X_train = X
    y_train = y
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # reshaping the data to 3d
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    callbacks = [EarlyStopping(monitor='val_loss', patience=100)]

    model.fit(X_train, y_train, validation_split=0.2, epochs=1000, callbacks=callbacks, batch_size=5, shuffle=False)
    model.save('models_with_tuner\\' + col_name + '_model_100.h5')
    print(' ')
    print('saved the model successfully')


df3 = pd.read_csv('pycharm_data.csv')
df3 = df3.drop(columns=['Unnamed: 0', 'date'])


for i in df3.columns:
    if i in ['west bengal']:
        predict_rem(i)
        time.sleep(15)
