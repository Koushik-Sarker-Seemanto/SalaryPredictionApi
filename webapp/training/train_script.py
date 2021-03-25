import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score

import os
settings_dir = os.path.dirname(__file__)
root = os.path.abspath(os.path.dirname(settings_dir))
read_path = os.path.join(root, 'dataset\\dataset_salary.csv')
write_path = os.path.join(root, 'training\\finalized_model.sav')


def train_ml_model():
    print('Reading Csv')
    data = pd.read_csv(read_path)
    data = data.dropna()
    ind_feat = data.iloc[:, :-1]
    dep_feat = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(ind_feat, dep_feat, test_size=0.2, random_state=0)
    regression = LinearRegression()
    print('Training Model')
    regression.fit(X_train, Y_train)

    prediction = regression.predict(X_test)
    score = r2_score(Y_test, prediction)

    print('r2_score: ', score)

    # save the model to disk
    pickle.dump(regression, open(write_path, 'wb'))
    print('Training Complete!!!')


train_ml_model()
