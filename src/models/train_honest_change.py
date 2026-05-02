import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

def train_honest_change(data_path, model_path):
    df = pd.read_csv(data_path)
    df = df.sample(n=min(1_000_000, len(df)), random_state=42)
    
    # Honest features: exclude sar_flood and sar_change (directly encode label)
    X = df[["sar_dry", "rainfall", "elevation"]]
    y = df["label"]
    
    print(f"Training honest change detection on {len(df)} samples")
    print(f"Features: {list(X.columns)}")
    print(f"Class balance: {(y==0).sum()} no-flood / {(y==1).sum()} flood")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=2, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=200, eval_metric='logloss', random_state=42)
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=1)
        print(f"\n{name}: AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        print(f"Feature importances:\n{importances.sort_values(ascending=False)}")
    
    # Save best
    best_model = models['XGBoost']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\nSaved: {model_path}")

if __name__ == "__main__":
    train_honest_change("data/processed/dataset_change_full.csv", 
                         "outputs/models/xgboost_honest_change.pkl")