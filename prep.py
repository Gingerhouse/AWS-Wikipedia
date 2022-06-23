import pandas as pd
import numpy as np
import os
import argparse

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    
    file = os.path.join(args.train, "wiki4HE.csv")
    data = pd.read_csv(file, sep = ';', engine="python")
    data = data.replace('?',np.nan)
    data = data.drop(columns = ['UNIVERSITY','UOC_POSITION','OTHERSTATUS','OTHER_POSITION', 'Vis2', 'PEU3', 'Im3']).dropna()

    nominal = ['GENDER', 'DOMAIN','PhD','USERWIKI']
    ordinal = ['PU1','PU2','PU3','PEU1',
         'PEU2','ENJ1','ENJ2','Qu1','Qu2','Qu3','Qu4','Qu5','Vis1','Vis3','Im1','Im2',
         'SA1','SA2','SA3','Use1','Use2','Use3','Use4','Use5','Pf1','Pf2','Pf3','JR1','JR2',
         'BI1','BI2','Inc1','Inc2','Inc3','Inc4','Exp1','Exp2','Exp3','Exp4','Exp5']

    ohe = OneHotEncoder(sparse = False)
    oe = OrdinalEncoder()

    ohe.fit_transform(data[nominal])
    oe.fit_transform(data[ordinal])

    X = data.drop(columns = ['PU3'])
    y = data.PU3
    best = SelectKBest(score_func=chi2, k=13).fit(X,y)

    cols = best.get_support(indices=True)
    X = data.iloc[:,cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


    model = RandomForestClassifier(max_depth = 7).fit(X_train,y_train)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model;

def predict_fn(input_data, model):
    return model.predict(input_data)