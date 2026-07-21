"""
train.py

Ağ trafiği anomali/saldırı tespiti için Dengesiz Sınıflara (Imbalanced Classes)
özel yapılandırılmış, temiz kod standartlarına uygun model eğitim betiği.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             precision_recall_curve)
from imblearn.ensemble import BalancedRandomForestClassifier

# ==========================================
# 1. YAPILANDIRMA (CONFIG)
# ==========================================
CONFIG = {
    "data_path": "data/processed/cicids2017_processed.csv",
    "model_dir": "models",
    "model_name": "random_forest.pkl",
    "metrics_name": "metrics.json",
    "random_state": 42,
    "test_size": 0.15,
    "val_size": 0.15,        # Kalan Train setinden Validation için ayrılacak oran
    "n_estimators": 500,      # Ağaç sayısı
    "n_jobs": -1              # Tüm CPU çekirdeklerini kullan
}

# ==========================================
# 2. LOGLAMA AYARLARI
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# 3. YARDIMCI FONKSİYONLAR
# ==========================================
def load_and_split_data():
    """Veriyi yükler; Train, Validation ve Test olarak 3 parçaya böler."""
    if not os.path.exists(CONFIG["data_path"]):
        logger.error(f"Veri seti bulunamadı: {CONFIG['data_path']}")
        raise FileNotFoundError(f"Lütfen verinin {CONFIG['data_path']} yolunda olduğundan emin olun.")

    logger.info("Veri seti yükleniyor...")
    df = pd.read_csv(CONFIG["data_path"])

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Adım 1: Test setini en sona saklamak üzere ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y
    )

    # Adım 2: Kalan veriyi Eğitim (Train) ve Doğrulama (Validation) olarak ayır
    # Matematiksel olarak doğru orantıyı tutturmak için val_ratio hesaplanıyor
    val_ratio = CONFIG["val_size"] / (1.0 - CONFIG["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=CONFIG["random_state"], stratify=y_temp
    )

    logger.info(f"Veri Bölünmesi Tamamlandı -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_best_threshold(y_true, y_probs):
    """Validation seti üzerinde F1 skorunu maksimize eden en iyi eşik (threshold) değerini bulur."""
    logger.info("Optimum threshold (eşik) değeri hesaplanıyor...")
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Sıfıra bölme hatasını önlemek için paydaya epsilon (1e-10) ekliyoruz
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])   
    best_threshold = thresholds[best_idx]

    logger.info(f"En iyi Threshold: {best_threshold:.4f} (Validation F1 Score: {f1_scores[best_idx]:.4f})")
    return best_threshold


# ==========================================
# 4. ANA EĞİTİM AKIŞI (MAIN)
# ==========================================
def main():
    # Modelin kaydedileceği klasörü oluştur
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    model_save_path = os.path.join(CONFIG["model_dir"], CONFIG["model_name"])
    metrics_save_path = os.path.join(CONFIG["model_dir"], CONFIG["metrics_name"])

    # Veriyi hazırla
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()

    # Model nesnesini oluştur
    model = BalancedRandomForestClassifier(
        n_estimators=CONFIG["n_estimators"],
        random_state=CONFIG["random_state"],
        n_jobs=CONFIG["n_jobs"]
    )

    # Modeli tüm Train seti üzerinde eğit (final model)
    logger.info("Balanced Random Forest modeli tüm Train seti üzerinde eğitiliyor...")
    model.fit(X_train, y_train)
    logger.info("Model eğitimi başarıyla tamamlandı.")

    # Train seti üzerinde performansı ölç (overfitting kontrolü için)
    train_preds = model.predict(X_train)
    train_f1 = f1_score(y_train, train_preds)
    logger.info(f"Train F1 (overfitting kontrolü): {train_f1:.4f}")

    # Threshold Optimizasyonu (Sadece Validation seti ile yapılıyor!)
    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, val_probs)

    # Test ve Değerlendirme (Modelin hiç görmediği Test seti ile yapılıyor!)
    logger.info("Test seti üzerinde nihai performans değerlendiriliyor...")
    test_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(int)

    test_f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": test_f1,
        "train_f1_score": train_f1,
        "best_threshold": float(best_threshold)
    }

    # Sonuçları terminale bas
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Train ve Test F1 arasındaki farkı karşılaştır (overfitting sinyali)
    gap = train_f1 - test_f1
    logger.info(f"Train F1: {train_f1:.4f}  |  Test F1: {test_f1:.4f}  |  Fark: {gap:.4f}")
    if gap > 0.05:
        logger.warning(
            "Train ve Test F1 arasındaki fark 0.05'ten büyük — model overfit "
            "etmiş olabilir. max_depth düşürmeyi veya min_samples_leaf'i "
            "artırmayı düşünebilirsiniz."
        )

    # Kaydetme İşlemleri
    logger.info("Model diske kaydediliyor: %s", model_save_path)
    joblib.dump(model, model_save_path)

    logger.info("Metrikler JSON olarak kaydediliyor: %s", metrics_save_path)
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Eğitim pipeline'ı sorunsuz şekilde tamamlandı!")


if __name__ == "__main__":
    main()