# src/models/spatial_cv.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def spatial_cv(data_path, n_splits=5):
    df = pd.read_csv(data_path)
    df = df.sample(n=min(1_000_000, len(df)), random_state=42)
    
    # Sort by elevation as proxy for spatial position (south = low, north = high)
    df = df.sort_values('elevation').reset_index(drop=True)
    
    fold_size = len(df) // n_splits
    scores = []
    scores_no_sar = []
    
    for i in range(n_splits):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_splits - 1 else len(df)
        
        test_mask = (df.index >= start) & (df.index < end)
        train_mask = ~test_mask
        
        # With SAR
        X_train = df.loc[train_mask, ["rainfall", "elevation", "flood_signal"]]
        y_train = df.loc[train_mask, "label"]
        X_test = df.loc[test_mask, ["rainfall", "elevation", "flood_signal"]]
        y_test = df.loc[test_mask, "label"]
        
        model = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=2, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        scores.append(auc)
        
        # Without SAR
        X_train_ns = df.loc[train_mask, ["rainfall", "elevation"]]
        X_test_ns = df.loc[test_mask, ["rainfall", "elevation"]]
        model_ns = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=2, random_state=42)
        model_ns.fit(X_train_ns, y_train)
        y_proba_ns = model_ns.predict_proba(X_test_ns)[:, 1]
        auc_ns = roc_auc_score(y_test, y_proba_ns)
        scores_no_sar.append(auc_ns)
        
        print(f"Fold {i+1}: elev {df.loc[start, 'elevation']:.1f} to {df.loc[end-1, 'elevation']:.1f} | AUC (all) = {auc:.4f} | AUC (no SAR) = {auc_ns:.4f}")
    
    print(f"\nSpatial CV AUC (all features): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"Spatial CV AUC (no SAR): {np.mean(scores_no_sar):.4f} (+/- {np.std(scores_no_sar):.4f})")
    
    # Feature importance
    model.fit(df[["rainfall", "elevation", "flood_signal"]], df["label"])
    importances = pd.Series(model.feature_importances_, index=["rainfall", "elevation", "flood_signal"])
    print(f"\nFeature importances:\n{importances.sort_values(ascending=False)}")

if __name__ == "__main__":
    spatial_cv("data/processed/dataset_final_balanced.csv")