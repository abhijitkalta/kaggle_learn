import pandas as pd

from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing


def run(fold):
    
    df = pd.read_csv('../input/cat_train_folds.csv')
    
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    
    
    for f in features:
        df.loc[:, f] = df[f].astype(str).fillna("NONE")
        
    
    df_train = df[df.kfold != fold ].reset_index(drop = True)
    df_valid = df[df.kfold == fold ].reset_index(drop = True)
    
    ohe = preprocessing.OneHotEncoder()
    
    full_data = pd.concat([df_train[features], df_valid[features]], axis = 0)
    
    ohe.fit(full_data[features])
    
    
    X_train = ohe.transform(df_train[features])
    y_train = df_train.target.values
    X_valid = ohe.transform(df_valid[features])
    y_valid = df_valid.target.values
    
    model = linear_model.LogisticRegression()
    
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_valid)[:, 1]
    
    auc = metrics.roc_auc_score(y_valid, y_probs)
    print(f"""
        Fold = {fold},
        Auc = {auc}
    """)
    
    
if __name__ == '__main__':
    run(0)
    run(1)
    run(2)
    run(3)