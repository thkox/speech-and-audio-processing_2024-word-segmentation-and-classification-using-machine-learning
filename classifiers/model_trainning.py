# import numpy as np
# from sklearn import svm
# from sklearn.linear_model import LinearRegression
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, SimpleRNN


# def train_classifiers():
    # Load the extracted features
    # X_train = np.load('X_train.npy')
    # Y_train = np.load('y_train.npy')
    #
    # # SVM Classifier
    # clf_svm = svm.SVC()
    # clf_svm.fit(X_train, Y_train)
    #
    # # Least Squares Classifier
    # clf_ls = LinearRegression()
    # clf_ls.fit(X_train, Y_train)
    #
    # # MLP Classifier
    # model_mlp = Sequential()
    # model_mlp.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
    # model_mlp.add(Dense(units=10, activation='softmax'))
    # model_mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model_mlp.fit(X_train, Y_train, validation_split=0.2, epochs=10)
    #
    # # RNN Classifier
    # model_rnn = Sequential()
    # model_rnn.add(SimpleRNN(units=64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # model_rnn.add(Dense(units=10, activation='softmax'))
    # model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model_rnn.fit(X_train, Y_train, validation_split=0.2, epochs=10)
