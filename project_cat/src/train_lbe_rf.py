import pandas as pd

from sklearn import metrics
from sklearn import ensemble
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv('../input/cat_train_folds.csv')
    
    features = [ f for f in df.columns if f not in ("id", "target", "kfold")]
    
    for cols in features:
        df.loc[:, cols] = df[cols].astype(str).fillna('NONE')
        
    for col in features:
        lbl = preprocessing.LabelEncoder()
        
        df.loc[:, col] = lbl.fit_transform(df[col])
    
    
    
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    X_train = df_train[features]
    X_valid = df_valid[features]
    
    y_train = df_train.target.values
    y_valid = df_valid.target.values
    
    model = ensemble.RandomForestClassifier(n_jobs = 1)
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_valid)[:, 1]
    
    auc = metrics.roc_auc_score(y_valid, y_probs)
    
    print(f"""
    Fold = {fold}, Auc = {auc}
    """)
    

if __name__ == '__main__':
    run(0)
    run(1)
    run(2)
    run(3)