@echo off
echo ===================================================
echo WEX 428 - Installing Requirements and Running Setup
echo ===================================================
pip install -r requirements.txt
pip install torch-directml
echo.
echo Building 2,500-image dataset and running experiments...
python experiment_module.py
echo.
echo Done! All plots and CSV metrics are saved in the results/ folder.
pause
