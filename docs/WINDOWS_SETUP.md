# RabitScal — Hướng Dẫn Dời Đô về Windows Xeon 95GB RAM

## Yêu cầu phần cứng
- **RAM**: 95GB (đủ cho 15 Optuna workers)
- **GPU**: GTX 1660 Super 6GB (XGBoost CUDA)
- **MT5**: MetaTrader 5 chạy native trên Windows

---

## Bước 1: Clone/Copy dự án

```powershell
# Copy toàn bộ folder RabitScal sang Windows
# Ví dụ: C:\RabitScal\
```

## Bước 2: Cài Python 3.12+

Tải từ [python.org](https://www.python.org/downloads/) → tick "Add to PATH"

```powershell
python --version   # Kiểm tra 3.12+
```

## Bước 3: Tạo venv + Cài dependencies

```powershell
cd C:\RabitScal
python -m venv venv
.\venv\Scripts\activate

# Core dependencies
pip install -r requirements.txt

# XGBoost CUDA (QUAN TRỌNG — bản có GPU support)
pip install xgboost --upgrade
```

> [!IMPORTANT]
> XGBoost 2.0+ tự detect CUDA nếu đã cài NVIDIA Driver + CUDA Toolkit.
> Không cần cài bản `xgboost-gpu` riêng nữa.

## Bước 4: Cài NVIDIA CUDA Toolkit

1. Tải [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) → cài
2. Kiểm tra:
```powershell
nvidia-smi              # Xem GPU info
python -c "import xgboost; print(xgboost.__version__)"
```

## Bước 5: Test GPU

```powershell
cd C:\RabitScal
python core/gpu_config.py
```

Kết quả mong đợi:
```
GPU:     NVIDIA GeForce GTX 1660 SUPER
VRAM:    6144 MB total
CUDA:    AVAILABLE
XGBoost GPU mode: tree_method=hist, device=cuda
GPU TRAINING TEST PASSED!
```

## Bước 6: Khởi động Socket Bridge

```powershell
cd C:\RabitScal
python socket_bridge.py --host 127.0.0.1
```

## Bước 7: MT5 EA

1. Copy `RabitScal_EA.mq5` vào MT5 Experts folder
2. Mở MetaEditor → Compile (F7)
3. MT5 → Tools → Options → Expert Advisors:
   - ☑ Allow algorithmic trading
   - ☑ Allow WebRequest → thêm `127.0.0.1`
4. Kéo EA vào chart → `InpServerIP = 127.0.0.1`

> [!TIP]
> Cùng máy = **không cần SSH tunnel** nữa! Localhost trực tiếp, zero latency.

## Bước 8: Chạy Training (Optuna)

```powershell
cd C:\RabitScal
python quant_main.py train --phase 8 --trials 2000 --workers 15
```

> [!CAUTION]
> **KHÔNG được dùng `--workers` > 15!** Windows `spawn` sẽ nhân bản RAM.
> Code đã có auto-clamp, nhưng tuyệt đối không bypass!

---

## Tổng kết thay đổi

| File | Thay đổi |
|------|----------|
| `socket_bridge.py` | Hardcode `127.0.0.1`, single socket port 15555 |
| `RabitScal_EA.mq5` | Default IP `127.0.0.1`, chunked transfer 8KB |
| `quant_main.py` | Workers default 50→15, auto-clamp max 15 |
| `core/gpu_config.py` | **[NEW]** XGBoost GPU: `tree_method=hist, device=cuda` |
