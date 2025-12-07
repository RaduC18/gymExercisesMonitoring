from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import butter, filtfilt
import joblib
from collections import deque
from threading import Timer, Lock
import tensorflow as tf


PROCESS_INTERVAL = 0.04

MIN_LEN = 60

DTW_THRESHOLD = 30.0

DEBUG = True


rf_model = joblib.load("random_forest_model.pkl")

dtw_reference = np.load("dtw_reference.npy")

ppg_model = tf.keras.models.load_model("ppg_cnn_lstm_model.h5")


app = Flask(__name__)

CORS(app)


buffer = deque()

buffer_lock = Lock()

ppg_buffer = deque(maxlen=185)


latest_result = {
    "final_result": -1,
    "dtw_result": -1,
    "rf_result": -1,
    "dtw_distance": -1.0,
    "consecutive_correct": 0,
    "consecutive_wrong": 0
}

consecutive_correct = 0

consecutive_wrong = 0


def butter_bandpass_filter(data, lowcut=0.5, highcut=20.0, fs=125.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def extract_features(sequence):
    arr = np.array(sequence)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)])


def resize_series_multidim(series, target_len):
    return np.array([
        np.interp(np.linspace(0, 1, target_len),
                  np.linspace(0, 1, len(series)),
                  series[:, i])
        for i in range(series.shape[1])
    ]).T


def analiza_repetare():
    global latest_result, consecutive_correct, consecutive_wrong

    with buffer_lock:
        if len(buffer) >= MIN_LEN:
            seq = np.array(buffer)
            accel_xyz = seq[:, :3]
            ref_xyz = dtw_reference[:, :3]
            target_len = min(len(accel_xyz), len(ref_xyz))
            seq_resized = resize_series_multidim(accel_xyz, target_len)
            ref_resized = resize_series_multidim(ref_xyz, target_len)

            try:
                dtw_dist, _ = fastdtw(seq_resized, ref_resized, dist=euclidean)
                dtw_result = int(dtw_dist < DTW_THRESHOLD)

                features = extract_features(seq).reshape(1, -1)
                rf_result = int(rf_model.predict(features).item())

                final_result = int(dtw_result == 1 and rf_result == 1)

                if final_result == 1:
                    consecutive_correct += 1
                    consecutive_wrong = 0
                else:
                    consecutive_wrong += 1
                    consecutive_correct = 0

                latest_result = {
                    "final_result": final_result,
                    "dtw_result": dtw_result,
                    "rf_result": rf_result,
                    "dtw_distance": float(dtw_dist),
                    "consecutive_correct": consecutive_correct,
                    "consecutive_wrong": consecutive_wrong
                }

                if DEBUG:
                    print(f"\n Processing: {len(seq)} samples")
                    print(f"DTW distance: {dtw_dist:.2f} → {dtw_result}")
                    print(f"RF result: {rf_result}")
                    print(f"Final: {'Correct' if final_result else 'Incorrect'}")
                    print(f"Correct consecutive: {consecutive_correct}")
                    print(f"Wrong consecutive: {consecutive_wrong}")

            except Exception as e:
                print("Analysis error:", e)

            finally:
                buffer.clear()

    Timer(PROCESS_INTERVAL, analiza_repetare).start()


@app.route("/predict_delta", methods=["POST"])
def predict_delta():
    try:
        data = request.get_json()
        sample = [float(data[k]) for k in ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']]
        with buffer_lock:
            buffer.append(sample)
            print(f"Buffer length: {len(buffer)}")
        return jsonify(latest_result)

    except Exception as e:
        print("Eroare generală:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/predict_ppg", methods=["POST"])
def predict_ppg():
    try:
        data = request.get_json()
        raw_val = data.get("ppgRaw", None)

        if raw_val is None or not isinstance(raw_val, (int, float)):
            return jsonify({"error": "Missing or invalid PPG value"}), 400

        prag_contact = 9000
        if raw_val < prag_contact:
            print(f"Value below the contact threshold ({raw_val}) — deletes all data.")
            ppg_buffer.clear()
            return jsonify({
                "ppg_result": -1,
                "confidence": 0.0,
                "ppg_score": 0.0
            })

        ppg_buffer.append(float(raw_val))

        if len(ppg_buffer) < 185:
            return jsonify({
                "ppg_result": -1,
                "confidence": 0.0,
                "ppg_score": None
            })

        signal_np = np.array(ppg_buffer, dtype=np.float32)
        std_raw = np.std(signal_np)

        if std_raw < 120:
            print(f"Weak signal - standard deviation ({std_raw:.2f}) is under 120.")
            ppg_buffer.clear()
            return jsonify({
                "ppg_result": -1,
                "confidence": 0.0,
                "ppg_score": 0.0
            })

        print(f"\n PPG raw (last 185): {np.round(signal_np, 2).tolist()}")

        signal_filtered = butter_bandpass_filter(signal_np, lowcut=0.5, highcut=20.0, fs=125.0, order=3)
        print(f"PPG filtered: {np.round(signal_filtered, 4).tolist()}")

        troughs = []
        adaptive_threshold = signal_filtered[0]
        last_trough_index = -np.inf
        ibi_limit = 0.6 * (125.0 / 2)

        for i in range(1, len(signal_filtered) - 1):
            if (signal_filtered[i - 1] > signal_filtered[i] < signal_filtered[i + 1] and
                    signal_filtered[i] < adaptive_threshold and
                    (i - last_trough_index) > ibi_limit):
                troughs.append((i, signal_filtered[i]))
                last_trough_index = i
                adaptive_threshold = signal_filtered[i] + 0.3 * abs(signal_filtered[i])

        print(f"Troughs detection: {len(troughs)} → {troughs}")

        signal_np_reshaped = signal_filtered.reshape(1, 185, 1)
        prediction = ppg_model.predict(signal_np_reshaped)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        ppg_score = round(confidence, 4)

        print(f"Prediction: {predicted_class} | IIR: {confidence:.4f} → Estimated PPG: {ppg_score}\n")

        ppg_buffer.clear()

        return jsonify({
            "ppg_result": predicted_class,
            "confidence": confidence,
            "ppg_score": ppg_score,
            "troughs": troughs
        })

    except Exception as e:
        print(" PPG Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Server running on http://192.168.1.119:5000")
    analiza_repetare()
    app.run(host="192.168.1.119", port=5000, debug=True)