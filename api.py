"""
api.py
Eğitilen makine öğrenmesi modelini dış dünyaya açan FastAPI uygulaması.
"""

import io
import json
import logging
import os
import subprocess
import tempfile

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.process_flow_meter import preprocess_flows

# --- LOGLAMA AYARLARI ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI uygulamasını başlat
app = FastAPI(
    title="Siber Saldırı Tespit API",
    description="Ağ trafiği verilerini analiz ederek Anomali/Saldırı tespiti yapar.",
    version="1.1"
)

# Global değişkenler (Uygulama kalkarken doldurulacak)
model = None
best_threshold = 0.5


# --- MODEL YÜKLEME (STARTUP) ---
@app.on_event("startup")
async def load_model_and_metrics():
    """API ayağa kalkarken modeli ve metrikleri belleğe yükler."""
    global model, best_threshold
    try:
        logger.info("Yapay Zeka Modeli belleğe yükleniyor...")
        model = joblib.load("models/random_forest.pkl")

        logger.info("Metrikler ve Threshold değeri yükleniyor...")
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
            best_threshold = metrics.get("best_threshold", 0.5)

        logger.info(f"Sistem hazır! Kullanılacak Threshold (Eşik) Değeri: {best_threshold:.4f}")
    except Exception as e:
        logger.error(f"Başlangıç sırasında kritik hata: {str(e)}")
        raise e


# --- TAHMİN (PREDICT) ENDPOINT'İ ---
@app.post("/predict")
async def predict_traffic(data: dict):
    """
    Ağ trafiği JSON verisini Body olarak alır ve tahmin sonucunu döner.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    try:
        input_df = pd.DataFrame([data])

        if 'Label' in input_df.columns:
            input_df = input_df.drop('Label', axis=1)

        attack_prob = model.predict_proba(input_df)[:, 1][0]
        is_attack = bool(attack_prob >= best_threshold)

        return {
            "status": "success",
            "prediction": "Attack" if is_attack else "Normal",
            "attack_probability": round(float(attack_prob), 4),
            "threshold_used": round(best_threshold, 4)
        }

    except Exception as e:
        logger.error(f"Tahmin sırasında hata oluştu: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Hatalı veri formatı: {str(e)}")


@app.post("/predict_pcap")
async def predict_pcap(file: UploadFile = File(...)):
    """
    Upload a PCAP/PCAPNG file, extract flows with CICFlowMeter,
    preprocess them, run the ML model, and return predictions.csv.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi.")

    if not file.filename.lower().endswith((".pcap", ".pcapng")):
        raise HTTPException(status_code=400, detail="Sadece .pcap veya .pcapng dosyaları kabul edilir.")

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(temp_dir, file.filename)
        flows_path = os.path.join(temp_dir, "flows.csv")

        # Save uploaded file
        try:
            with open(input_path, "wb") as f:
                f.write(await file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dosya kaydedilemedi: {str(e)}")

        # Run CICFlowMeter
        try:
            result = subprocess.run(
                ["cicflowmeter", "-f", input_path, "-c", flows_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            logger.info(f"CICFlowMeter stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"CICFlowMeter failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"CICFlowMeter hatası: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="CICFlowMeter zaman aşımına uğradı.")

        if not os.path.exists(flows_path):
            raise HTTPException(
                status_code=500,
                detail="CICFlowMeter çalıştı ama flows.csv üretilmedi. "
                       "Pcap dosyasında yeterli paket/flow olmayabilir."
            )

        # Preprocess: remap columns, keep identifiers, clean inf/NaN
        try:
            df, identifiers, n_dropped = preprocess_flows(flows_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail="Temizlemeden sonra tahmin edilebilecek geçerli flow kalmadı."
            )

        # Predict
        try:
            probs = model.predict_proba(df)[:, 1]
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model tahmini başarısız: {str(e)}")

        predictions = ["Attack" if p >= best_threshold else "Normal" for p in probs]

        # Build result, keeping flow identifiers if we have them
        if identifiers is not None:
            result_df = identifiers.copy()
            result_df["Prediction"] = predictions
            result_df["Attack Probability"] = probs.round(4)
        else:
            result_df = pd.DataFrame({
                "Prediction": predictions,
                "Attack Probability": probs.round(4)
            })

        if n_dropped:
            logger.warning(f"Dropped {n_dropped} flow(s) with invalid/inf values before prediction")

        # --- KEY FIX ---
        # Read the CSV into memory and return it directly, instead of
        # FileResponse pointing at a path inside temp_dir. FileResponse
        # streams the file AFTER this function returns, but temp_dir is
        # deleted the instant the `with` block exits (i.e. on return),
        # causing a FileNotFoundError race. Returning bytes directly
        # avoids the race entirely.
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")

        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )


# --- UYGULAMAYI ÇALIŞTIRMA ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)