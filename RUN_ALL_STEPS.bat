@echo off
echo ================================================================================
echo GUJARAT CROP RECOMMENDATION SYSTEM - COMPLETE SETUP
echo ================================================================================

echo.
echo [STEP 1] Setting up Python Virtual Environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [STEP 2] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo [STEP 3] Installing Required Packages...
pip install -r requirements.txt

echo.
echo [STEP 4] Setting up Database...
python db\create_tables.py

echo.
echo [STEP 5] Training/Loading ML Models...
python model\Amodel.py

echo.
echo [STEP 6] Starting the Application...
echo The application will start on http://localhost:8501
echo Press Ctrl+C to stop the application
streamlit run navigation.py

echo.
echo ================================================================================
echo SETUP COMPLETE!
echo ================================================================================
pause