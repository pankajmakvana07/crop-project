#!/bin/bash
# Complete Setup and Run Script for Linux/Mac
# This script runs all necessary steps to set up and run the application

echo "================================================================================"
echo "CROP RECOMMENDATION SYSTEM - COMPLETE SETUP"
echo "================================================================================"
echo ""

# Step 1: Check if models exist
echo "[STEP 1/4] Checking if models exist..."
if [ -f "model/crop_recommendation_models.pkl" ]; then
    echo "✓ Models found! Skipping training."
    echo ""
else
    echo "✗ Models not found. Training required."
    echo ""
    
    # Step 2: Train models
    echo "[STEP 2/4] Training models (this may take 5-10 minutes)..."
    echo ""
    cd model
    python Amodel.py
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Model training failed!"
        echo "Please check the error messages above."
        exit 1
    fi
    cd ..
    echo ""
    echo "✓ Models trained successfully!"
    echo ""
fi

# Step 3: Test models
echo "[STEP 3/4] Testing models..."
echo ""
cd model
python test_models_simple.py
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Some tests failed, but continuing..."
fi
cd ..
echo ""

# Step 4: Run application
echo "[STEP 4/4] Starting application..."
echo ""
echo "================================================================================"
echo "APPLICATION STARTING"
echo "================================================================================"
echo ""
echo "The application will open in your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""
echo "================================================================================"
echo ""

streamlit run navigation.py
