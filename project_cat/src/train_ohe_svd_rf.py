import pandas as pd

from sklearn import metrics
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import decomposition

from scipy import sparse


def run(fold):
    df = pd.read_csv('../input/cat_train_folds.csv')
    
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
        
    
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    
    ohe = preprocessing.OneHotEncoder()
    
    full_data = pd.concat([df_train[features], df_valid[features]], axis = 0)
    
    ohe.fit(full_data)
    
    X_train = ohe.transform(df_train[features])
    y_train = df_train.target.values
    
    X_valid = ohe.transform(df_valid[features])
    y_valid = df_valid.target.values
    
    
    svd = decomposition.TruncatedSVD(n_components = 20)
    
    full_sparse = sparse.vstack((X_train, X_valid))
    
    svd.fit(full_sparse)
    
    X_train = svd.transform(X_train)
    X_valid = svd.transform(X_valid)
    
    model = ensemble.RandomForestClassifier(n_jobs = 1)
    
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_valid)[:, 1]
    
    auc = metrics.roc_auc_score(y_valid, y_probs)
    
    print(f"""
    Fold = {fold}, Auc = {auc}
    """)
    
if __name__ == '__main__':
    run(0)
    run(3)