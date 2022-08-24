import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


def run(fold):
    
    df = pd.read_csv('../input/cat_train_folds.csv')
    
    features = [f for f in df.columns if f not in ('id', 'target', 'kfold')]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
        
    
    for col in features:
        le = preprocessing.LabelEncoder()
        
        df.loc[:, col] = le.fit_transform(df[col])
        
    
    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    
    X_train = df_train[features]
    X_valid = df_valid[features]
    
    y_train = df_train.target.values
    y_valid = df_valid.target.values
    
    model = xgb.XGBClassifier(
        n_jobs = 1,
        max_depth = 7,
        n_estimators = 200
    )
    
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_valid)[:, 1]
    
    auc = metrics.roc_auc_score(y_valid, y_probs)
    
    print(f"""
    Fold = {fold}, Auc = {auc}
    """)
    

if __name__ == '__main__':
    run(0)
    run(4)
    
    
    