import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

def train_change_model(data_path, model_path):
    df = pd.read_csv(data_path)
    df = df.sample(n=min(2_000_000, len(df)), random_state=42)
    
    # Key features: SAR change magnitude, dry season baseline, flood season
    X = df[["sar_dry", "sar_flood", "sar_change"]]
    y = df["label"]
    
    print(f"Training change detection on {len(df)} samples")
    print(f"Features: {list(X.columns)}")
    print(f"Class balance: {(y==0).sum()} no-flood / {(y==1).sum()} flood")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        scale_pos_weight=50,  # Flood pixels will be much rarer now
        eval_metric='logloss',
        random_state=42
    )
    
    scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=2)
    print(f"\nAUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    model.fit(X, y)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nSaved: {model_path}")

if __name__ == "__main__":
    train_change_model("data/processed/dataset_change.csv", "outputs/models/xgboost_change.pkl")