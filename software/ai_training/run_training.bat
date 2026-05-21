@echo off
cd /d "%~dp0"

echo ==============================
echo Chess AI Training Start
echo ==============================

python train_model_local.py

echo ==============================
echo Random Image Evaluation Start
echo ==============================

python evaluate_random_images.py

echo ==============================
echo Training Cycle Finished
echo ==============================

pause
