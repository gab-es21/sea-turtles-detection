Write-Host "====================================="
Write-Host " YOLOv9 Training + TensorBoard"
Write-Host "====================================="

# 1. Activate venv
Write-Host "[1/4] Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# 2. Start TensorBoard in background
Write-Host "[2/4] Starting TensorBoard..."
Start-Process powershell -ArgumentList `
    "-NoExit", `
    "-Command", `
    "cd src; tensorboard --logdir runs/train --port 6006"

# Give TensorBoard time to start
Start-Sleep -Seconds 5

# 3. Open browser
Write-Host "[3/4] Opening TensorBoard in browser..."
Start-Process "http://localhost:6006"

# 4. Start training (foreground)
Write-Host "[4/4] Starting YOLOv9 training..."
python scripts\train_yolov9.py

Write-Host "Training finished."
