#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  EV05d LAUNCH — $200 Live Rehearsal (Sếp Vũ Approved)
#  5 mã × 3 TF = 15 studies × 2000 trials = 30,000 kịch bản
# ═══════════════════════════════════════════════════════════════

set -e
cd /home/xeon-sever/RabitScal

echo "🧹 Step 1: Dọn sạch DB cũ (ev05c + ev05d)..."
rm -f data/optuna_ev05c_*.db
rm -f data/optuna_ev05d_*.db
echo "   ✅ Đã xóa toàn bộ DB cũ"

echo ""
echo "🔥 Step 2: Launch Ev05d training (nohup background)..."
nohup venv/bin/python quant_main.py train --phase 8 --trials 2000 --workers 40 \
  --spread 0.00015 --out logs/An_Latest_Report.md > /tmp/ev05d_train.log 2>&1 &
TRAIN_PID=$!
echo "   ✅ Training PID: $TRAIN_PID"
echo "   📄 Log: /tmp/ev05d_train.log"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BÓP CÒ! Ev05d đang chạy nền (PID: $TRAIN_PID)"
echo "  Monitor:  tail -f /tmp/ev05d_train.log"
echo "  Check DB: ls -lh data/optuna_ev05d_*.db"
echo "═══════════════════════════════════════════════════════════════"
