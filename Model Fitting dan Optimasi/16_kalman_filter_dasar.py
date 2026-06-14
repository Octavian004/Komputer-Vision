"""
==========================================================================
PERCOBAAN 16: KALMAN FILTER UNTUK ESTIMASI STATE
==========================================================================
Program ini mempelajari kalman filter untuk estimasi state.
Praktikum 16 - Kalman Filter untuk Estimasi State
Modul 05: Model Fitting dan Optimasi

Topik: cv2.KalmanFilter, state estimation, prediction + correction
Referensi: Learning OpenCV Ch.15 (Bradski & Kaehler),
           Machine Learning for OpenCV Ch.9 (Beyeler)

Hasil: Visualisasi dan analisis disimpan ke folder output/
==========================================================================
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Mendapatkan direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mendefinisikan path folder gambar input
IMAGE_DIR = os.path.join(SCRIPT_DIR, "image")

# Mendefinisikan path folder output untuk menyimpan hasil
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Membuat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)



def demo_1d_kalman():
    """Kalman Filter 1D — pelacakan nilai skalar bernoisy."""
    np.random.seed(0)
    t = np.linspace(0, 4 * np.pi, 100)
    true_signal = np.sin(t)
    noisy_signal = true_signal + np.random.normal(0, 0.5, len(t))

    # ---- Kalman Filter manual (1D) ----
    # State: posisi x
    # Transition: x_k = x_{k-1}  (constant model)
    # Measurement: z_k = x_k + noise
    x_est = 0.0         # estimasi awal
    P = 1.0             # error covariance awal
    Q = 1e-4            # process noise variance (kecil = smooth)
    R = 0.25            # measurement noise variance (≈ 0.5^2)

    estimates = []
    for z in noisy_signal:
        # Prediction
        x_pred = x_est
        P_pred = P + Q
        # Update (correction)
        K = P_pred / (P_pred + R)     # Kalman Gain
        x_est = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        estimates.append(x_est)

    estimates = np.array(estimates)
    print(f"[1D Kalman] MSE noisy: {np.mean((noisy_signal - true_signal)**2):.4f}")
    print(f"[1D Kalman] MSE Kalman: {np.mean((estimates - true_signal)**2):.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(t, true_signal, 'g-', linewidth=2, label='True Signal')
    plt.plot(t, noisy_signal, 'r.', alpha=0.5, markersize=4, label='Noisy Measurement')
    plt.plot(t, estimates, 'b-', linewidth=2, label='Kalman Estimate')
    plt.legend(); plt.title("Kalman Filter 1D"); plt.grid(True)
    plt.tight_layout(); plt.savefig("output_16_kalman_1d.png", dpi=100); plt.show()


def demo_cv2_kalman_filter_2d():
    """cv2.KalmanFilter untuk tracking posisi 2D (constant velocity model)."""
    # Model: state = [x, y, vx, vy], measurement = [x, y]
    kf = cv2.KalmanFilter(4, 2)

    # Transition matrix A: [1,0,dt,0; 0,1,0,dt; 0,0,1,0; 0,0,0,1]
    dt = 1.0
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float32)

    # Measurement matrix H: mengamati posisi saja
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    # Trajectory: lingkaran dengan noise
    np.random.seed(42)
    n = 80
    angles = np.linspace(0, 2 * np.pi, n)
    true_traj = np.column_stack([150 + 100 * np.cos(angles),
                                  150 + 100 * np.sin(angles)])
    noisy_traj = true_traj + np.random.normal(0, 8, true_traj.shape)

    kf.statePre = np.array([true_traj[0, 0], true_traj[0, 1], 0, 0], dtype=np.float32).reshape(4, 1)

    predictions, corrections = [], []
    for meas in noisy_traj:
        pred = kf.predict()
        corr = kf.correct(meas.astype(np.float32).reshape(2, 1))
        predictions.append(pred[:2].flatten())
        corrections.append(corr[:2].flatten())

    predictions = np.array(predictions)
    corrections = np.array(corrections)

    err_noisy = np.mean(np.linalg.norm(noisy_traj - true_traj, axis=1))
    err_kalman = np.mean(np.linalg.norm(corrections - true_traj, axis=1))
    print(f"[2D Kalman] Error rata-rata tanpa Kalman: {err_noisy:.2f}px")
    print(f"[2D Kalman] Error rata-rata dengan Kalman: {err_kalman:.2f}px")

    plt.figure(figsize=(8, 8))
    plt.plot(true_traj[:, 0], true_traj[:, 1], 'g-', linewidth=2, label='Ground Truth')
    plt.plot(noisy_traj[:, 0], noisy_traj[:, 1], 'r.', alpha=0.4, s=30, label='Noisy Measurement')
    plt.scatter(corrections[:, 0], corrections[:, 1], c='blue', s=15, alpha=0.7, label='Kalman Corrected')
    plt.legend(); plt.title("Kalman Filter 2D — Tracking Lingkaran"); plt.axis('equal'); plt.grid(True)
    plt.tight_layout(); plt.savefig("output_16_kalman_2d.png", dpi=100); plt.show()
    return kf


def demo_kalman_prediction_missing():
    """Demonstrasi prediksi saat measurement hilang (occlusion simulation)."""
    kf = cv2.KalmanFilter(4, 2)
    dt = 1.0
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
    kf.statePost = np.array([50, 150, 5, -2], dtype=np.float32).reshape(4, 1)

    np.random.seed(7)
    n = 60
    true_x = 50 + 5 * np.arange(n) + np.zeros(n)
    true_y = 150 - 2 * np.arange(n)
    noise = np.random.normal(0, 5, (n, 2))
    meas_x = true_x + noise[:, 0]
    meas_y = true_y + noise[:, 1]

    # Simulasikan occlusion: no measurement antara frame 25-40
    occlusion_start, occlusion_end = 25, 40

    track_x, track_y, mode_list = [], [], []
    for i in range(n):
        pred = kf.predict()
        if i < occlusion_start or i >= occlusion_end:
            # measurement tersedia
            z = np.array([[meas_x[i]], [meas_y[i]]], dtype=np.float32)
            state = kf.correct(z)
            mode_list.append('measure')
        else:
            # occlusion: hanya prediksi
            state = pred
            mode_list.append('predict')
        track_x.append(state[0, 0])
        track_y.append(state[1, 0])

    plt.figure(figsize=(12, 5))
    cmap_color = ['blue' if m == 'measure' else 'orange' for m in mode_list]
    plt.plot(true_x, true_y, 'g-', linewidth=2, label='Ground Truth')
    plt.scatter(meas_x[:occlusion_start], meas_y[:occlusion_start], c='lightblue', s=25, alpha=0.7)
    plt.scatter(meas_x[occlusion_end:], meas_y[occlusion_end:], c='lightblue', s=25, alpha=0.7, label='Measurements')
    plt.plot(track_x, track_y, 'b-', linewidth=1.5, label='Kalman Track')
    pred_x = [track_x[i] for i in range(n) if mode_list[i] == 'predict']
    pred_y = [track_y[i] for i in range(n) if mode_list[i] == 'predict']
    plt.scatter(pred_x, pred_y, c='orange', s=40, zorder=5, label=f'Prediksi (frame {occlusion_start}-{occlusion_end})')
    plt.legend(); plt.title("Kalman Filter — Prediksi saat Occlusion"); plt.grid(True)
    plt.tight_layout(); plt.savefig("output_16_kalman_occlusion.png", dpi=100); plt.show()
    print(f"  Frame {occlusion_start}-{occlusion_end}: hanya prediksi tanpa measurement (occlusion)")


def demo_kalman_parameters():
    """Pengaruh Q (process noise) dan R (measurement noise)."""
    np.random.seed(0)
    t = np.linspace(0, 4 * np.pi, 100)
    true_sig = np.sin(t)
    noisy = true_sig + np.random.normal(0, 0.4, len(t))

    configs = [
        ("Q kecil, R kecil (percaya meas.)", 1e-5, 0.04),
        ("Q kecil, R besar (percaya model)", 1e-5, 10.0),
        ("Q besar, R kecil (adaptif cepat)", 1.0, 0.04),
        ("Q=R seimbang", 0.1, 0.16),
    ]

    plt.figure(figsize=(16, 8))
    for idx, (label, Q, R) in enumerate(configs):
        x_est = 0.0; P = 1.0
        ests = []
        for z in noisy:
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x_est = x_est + K * (z - x_est)
            P = (1 - K) * P_pred
            ests.append(x_est)
        plt.subplot(2, 2, idx + 1)
        plt.plot(t, true_sig, 'g-', linewidth=1.5, label='True')
        plt.plot(t, noisy, 'r.', alpha=0.3, markersize=3, label='Noisy')
        plt.plot(t, ests, 'b-', linewidth=1.5, label='Kalman')
        plt.title(f"{label}\nQ={Q}, R={R}")
        mse = np.mean((np.array(ests) - true_sig) ** 2)
        plt.legend(fontsize=7); plt.grid(True)
        plt.xlabel(f"MSE={mse:.4f}")
    plt.suptitle("Pengaruh Parameter Q dan R pada Kalman Filter")
    plt.tight_layout(); plt.savefig("output_16_kalman_params.png", dpi=100); plt.show()


if __name__ == "__main__":
    print("=" * 55)
    print("PRAKTIKUM 16: KALMAN FILTER UNTUK ESTIMASI STATE")
    print("=" * 55)

    print("\n[1] Kalman Filter 1D — Dasar")
    demo_1d_kalman()

    print("\n[2] cv2.KalmanFilter 2D — Tracking Posisi")
    demo_cv2_kalman_filter_2d()

    print("\n[3] Prediksi saat Occlusion (Measurement Hilang)")
    demo_kalman_prediction_missing()

    print("\n[4] Pengaruh Parameter Q dan R")
    demo_kalman_parameters()

    print("\n[SELESAI] Semua demo Kalman Filter berhasil dijalankan.")
