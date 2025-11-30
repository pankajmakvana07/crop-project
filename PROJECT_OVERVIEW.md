# Gujarat Crop Recommendation System - Project Overview

## 📋 Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Database Schema](#database-schema)
- [API Integration](#api-integration)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🌾 Introduction

The **Gujarat Crop Recommendation System** is an AI-powered web application designed to help farmers and agricultural professionals make informed decisions about crop selection based on soil conditions, climate data, and regional agricultural patterns specific to Gujarat, India.

### Problem Statement
Farmers often struggle to determine which crops are most suitable for their land, leading to:
- Suboptimal crop selection
- Lower yields
- Economic losses
- Inefficient resource utilization

### Solution
Our system uses machine learning models trained on 17,818 agricultural data points from Gujarat to provide:
- **Accurate crop recommendations** based on soil and environmental conditions
- **Yield predictions** for better planning
- **Treatment plans** with fertilizer and pesticide recommendations
- **Multi-language support** for accessibility

---

## ✨ Features

### Core Features
1. **Normal Prediction Mode**
   - Get top 3 crop recommendations
   - Probability-based ranking
   - Suitability confidence scores
   - Yield predictions for each crop

2. **Advanced Prediction Mode**
   - Check suitability for specific crops
   - Detailed analysis with reasoning
   - Domain knowledge integration
   - Treatment recommendations

3. **User Management**
   - Secure authentication with bcrypt
   - Email and phone number login
   - OTP verification
   - Guest mode access

4. **Soil Data Management**
   - Add/Edit soil details
   - District and Taluka selection
   - Soil type and pH tracking
   - Historical data storage

5. **Prediction History**
   - Track all predictions
   - View past recommendations
   - Statistics and analytics
   - Export capabilities

6. **Multi-language Support**
   - Translation to 12 Indian languages
   - Hindi, Gujarati, Marathi, Tamil, Telugu, etc.
   - Real-time translation of results

### Advanced Features
- **Domain Knowledge Rules**: pH and soil type validation
- **Realistic Yield Ranges**: Crop-specific yield constraints
- **Probability Calibration**: Isotonic regression for accurate confidence
- **Treatment Database**: 21 crops with detailed cultivation plans
- **Responsive UI**: Modern card-based design
- **Data Visualization**: Charts and metrics

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                    (Streamlit Web App)                       │
├─────────────────────────────────────────────────────────────┤
│  Login/Signup  │  Dashboard  │  Soil Details  │  Prediction │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────────────────┐
         │                                                   │
┌────────▼────────┐                              ┌──────────▼─────────┐
│  Authentication │                              │   ML Model Layer   │
│     Layer       │                              │                    │
├─────────────────┤                              ├────────────────────┤
│ • JWT Tokens    │                              │ • XGBoost Crop     │
│ • Bcrypt Hash   │                              │ • Random Forest    │
│ • OTP Service   │                              │ • Gradient Boost   │
│ • Cookie Mgmt   │                              │ • SMOTE Balancing  │
└────────┬────────┘                              └──────────┬─────────┘
         │                                                   │
         │                                                   │
┌────────▼────────────────────────────────────────────────┬─▼─────────┐
│              Database Layer (PostgreSQL)                │           │
├─────────────────────────────────────────────────────────┤           │
│ • users          • soil_details    • prediction_history │           │
└─────────────────────────────────────────────────────────┴───────────┘
```

### Data Flow

1. **User Registration/Login**
   ```
   User → Authentication → JWT Token → Session Management
   ```

2. **Soil Data Entry**
   ```
   User Input → Validation → Database Storage → Confirmation
   ```

3. **Crop Prediction**
   ```
   User Request → Fetch Soil Data → ML Model Processing → 
   Domain Rules Application → Result Formatting → Display
   ```

---

## 💻 Technology Stack

### Frontend
- **Streamlit** (v1.x): Web application framework
- **Streamlit Components**: Enhanced UI elements
- **CSS3**: Custom styling and animations
- **JavaScript**: Cookie management

### Backend
- **Python 3.8+**: Core programming language
- **PostgreSQL**: Relational database
- **psycopg2**: Database adapter
- **bcrypt**: Password hashing
- **PyJWT**: Token authentication

### Machine Learning
- **scikit-learn**: Model training and preprocessing
- **XGBoost**: Gradient boosting classifier
- **imbalanced-learn**: SMOTE for class balancing
- **pandas**: Data manipulation
- **numpy**: Numerical computations

### Additional Libraries
- **deep-translator**: Multi-language support
- **yagmail**: Email notifications
- **python-dotenv**: Environment management
- **matplotlib/seaborn**: Data visualization
- **SHAP**: Model explainability

---

## 📁 Project Structure

```
project/
├── model/                          # ML models and training
│   ├── Amodel.py                  # Original model training script
│   ├── improved_Amodel.py         # Enhanced model with fixes
│   ├── crop_recommendation_models.pkl  # Trained models
│   ├── crop_treatments.json       # Treatment database
│   ├── gujarat_full_crop_dataset.csv  # Training dataset (17,818 records)
│   ├── plots/                     # Model visualizations
│   └── test_*.py                  # Testing scripts
│
├── page/                          # UI pages
│   ├── app.py                     # Main dashboard
│   ├── login_page.py              # Authentication
│   ├── signup_page.py             # User registration
│   ├── AddDetails.py              # Soil data management
│   ├── pridect_dynamic.py         # Prediction interface
│   ├── prediction_history.py      # History viewer
│   ├── Translate.py               # Translation utilities
│   └── taluka.csv                 # Gujarat location data
│
├── utils/                         # Utility modules
│   ├── model_integration.py       # ML model wrapper
│   ├── db_handler.py              # Database operations
│   ├── auth_token.py              # JWT authentication
│   ├── otp_handler.py             # OTP service
│   └── init_session.py            # Session management
│
├── db/                            # Database scripts
│   ├── create_db.py               # Database initialization
│   ├── delete_db.py               # Database cleanup
│   └── add_prediction_history_table.py  # Schema updates
│
├── navigation.py                  # Main app router
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── README.md                      # Quick start guide
├── PROJECT_OVERVIEW.md            # This file
├── MODEL_IMPROVEMENTS.md          # Model enhancement docs
├── LICENSE                        # MIT License
└── RUN_ALL_STEPS.*               # Setup scripts
```

---

## 🤖 Machine Learning Models

### 1. Crop Classification Model (XGBoost)
**Purpose**: Predict top crop recommendations

**Architecture**:
```python
XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

**Features** (52 total):
- Soil parameters: pH, EC, organic carbon, NPK
- Climate data: temperature, rainfall, humidity
- Location: district, taluka (encoded)
- Engineered features: NPK ratios, pH distances, season indicators

**Performance**:
- Accuracy: 87.51%
- Macro F1-Score: 81.48%
- Top-3 Accuracy: 99.47%

### 2. Suitability Model (Random Forest + Calibration)
**Purpose**: Determine if a crop can grow in given conditions

**Architecture**:
```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=5,
    class_weight='balanced'
) + CalibratedClassifierCV(method='isotonic')
```

**Performance**:
- Accuracy: 89.96%
- Precision: 89.80%
- Recall: 98.67%
- ROC AUC: 86.72%

### 3. Yield Prediction Model (Gradient Boosting)
**Purpose**: Estimate crop yield in quintal/hectare

**Architecture**:
```python
GradientBoostingRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    min_samples_split=15,
    min_samples_leaf=5,
    subsample=0.8
)
```

**Performance**:
- RMSE: 0.14
- MAE: 0.02
- R² Score: 1.00

### Feature Engineering

**Temperature Features**:
- `temp_range`: Max - Min temperature
- `temp_optimal_rice`: Distance from 25°C
- `temp_optimal_wheat`: Distance from 20°C

**NPK Features**:
- `NPK_ratio_NP`: Nitrogen/Phosphorus ratio
- `NPK_ratio_NK`: Nitrogen/Potassium ratio
- `NPK_sum`: Total NPK content
- `NPK_balance`: Nitrogen proportion

**pH Features**:
- `pH_squared`: Non-linear pH effect
- `pH_optimal_rice`: Distance from pH 6.5
- `pH_optimal_wheat`: Distance from pH 6.8
- `pH_acidic`: Binary flag (pH < 6.0)
- `pH_alkaline`: Binary flag (pH > 8.0)

**Climate Interactions**:
- `moisture_temp_interaction`: Soil moisture × Temperature
- `rainfall_humidity_interaction`: Rainfall × Humidity
- `water_stress_indicator`: ET / (Rainfall + 1)

**Temporal Features**:
- `month_sin`, `month_cos`: Cyclical month encoding
- `is_kharif`: Kharif season indicator (June-Oct)
- `is_rabi`: Rabi season indicator (Nov-March)

### Domain Knowledge Rules

The system integrates agricultural expertise through rule-based adjustments:

**Rice Suitability**:
- Ideal: Clay/Loamy soil + pH 5.5-7.0 → +30% confidence
- Good: Clay/Loamy soil + pH 5.0-7.5 → +20% confidence
- Not Suitable: pH outside 4.5-8.0 → -20% penalty

**Wheat Suitability**:
- Ideal: Loamy/Clay soil + pH 6.0-7.5 → +30% confidence
- Good: Loamy/Clay soil + pH 5.5-8.0 → +20% confidence

**Cotton Suitability**:
- Ideal: Black Cotton soil + pH 6.5-8.0 → +30% confidence
- Good: Black Cotton soil + pH 6.0-8.5 → +20% confidence

### Realistic Yield Ranges

Crop-specific yield constraints (quintal/hectare):

| Crop | Min | Typical Low | Typical High | Max |
|------|-----|-------------|--------------|-----|
| Rice | 15 | 25 | 40 | 50 |
| Wheat | 20 | 25 | 35 | 45 |
| Cotton | 10 | 15 | 25 | 35 |
| Bajra | 10 | 13 | 20 | 25 |
| Groundnut | 12 | 17 | 25 | 30 |
| Sugarcane | 600 | 700 | 850 | 1000 |
| Potato | 150 | 200 | 250 | 300 |

---

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hash_password TEXT NOT NULL,
    username TEXT NOT NULL,
    PhoneNumber BIGINT NOT NULL
);
```

### Soil Details Table
```sql
CREATE TABLE soil_details (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    state VARCHAR(255) NOT NULL,
    district VARCHAR(255) NOT NULL,
    taluka VARCHAR(255) NOT NULL,
    soil_type VARCHAR(255) NOT NULL,
    pH NUMERIC(4,2) NOT NULL
);
```

### Prediction History Table
```sql
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    prediction_type VARCHAR(50) NOT NULL,
    district VARCHAR(255) NOT NULL,
    taluka VARCHAR(255) NOT NULL,
    soil_type VARCHAR(255) NOT NULL,
    soil_ph NUMERIC(4,2) NOT NULL,
    predicted_crop VARCHAR(255),
    prediction_result TEXT NOT NULL,
    confidence NUMERIC(5,2),
    predicted_yield NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔌 API Integration

### Model Integration API

```python
from utils.model_integration import get_prediction_service

# Initialize service
service = get_prediction_service()

# Normal prediction
result = service.predict_normal(user_id, top_k=3)

# Advanced prediction
result = service.predict_advanced(user_id, crop_name="Rice")

# Get available crops
crops = service.get_available_crops()
```

### Response Format

**Normal Prediction**:
```json
{
  "user_id": 1,
  "district": "SURAT",
  "taluka": "Palsana",
  "soil_type": "Clay",
  "soil_pH": 6.5,
  "model": "normal",
  "top_recommendations": [
    {
      "crop": "Rice",
      "probability": 48.5,
      "suitability": "Yes",
      "suitability_confidence": 85.2,
      "predicted_yield_quintal_per_ha": 35.8,
      "treatment": { ... }
    }
  ]
}
```

**Advanced Prediction**:
```json
{
  "user_id": 1,
  "district": "SURAT",
  "taluka": "Palsana",
  "soil_type": "Clay",
  "soil_pH": 6.5,
  "model": "advanced",
  "crop": "Rice",
  "prediction": "Grow",
  "confidence": 85.2,
  "reason": "Ideal: Clay soil with pH 6.5 is excellent for Rice cultivation",
  "predicted_yield_quintal_per_ha": 35.8,
  "treatment": { ... }
}
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env with your database credentials and API keys
```

### Step 5: Initialize Database
```bash
python db/create_db.py
```

### Step 6: Train Models (if needed)
```bash
python model/improved_Amodel.py
```

### Step 7: Run Application
```bash
streamlit run navigation.py
```

---

## 📖 Usage

### For Farmers

1. **Register/Login**
   - Create account with email/phone
   - Verify with OTP

2. **Add Soil Details**
   - Select district and taluka
   - Enter soil type and pH
   - Save for future predictions

3. **Get Recommendations**
   - Choose Normal mode for top 3 crops
   - Or Advanced mode for specific crop
   - View treatment plans

4. **Track History**
   - Review past predictions
   - Compare recommendations
   - Export data

### For Developers

1. **Model Retraining**
   ```bash
   python model/improved_Amodel.py
   ```

2. **Testing**
   ```bash
   python model/test_improved_predictions.py
   ```

3. **Database Management**
   ```bash
   python db/create_db.py  # Initialize
   python db/delete_db.py  # Reset
   ```

---

## 📊 Model Performance

### Classification Metrics
- **Accuracy**: 87.51%
- **Macro F1-Score**: 81.48%
- **Top-3 Accuracy**: 99.47%
- **Cross-Validation**: 85.3% ± 2.1%

### Suitability Metrics
- **Accuracy**: 89.96%
- **Precision**: 89.80%
- **Recall**: 98.67%
- **ROC AUC**: 86.72%

### Yield Prediction Metrics
- **RMSE**: 0.14 quintal/ha
- **MAE**: 0.02 quintal/ha
- **R² Score**: 1.00

### Dataset Statistics
- **Total Records**: 17,818
- **Unique Crops**: 21
- **Districts Covered**: All Gujarat districts
- **Talukas**: 250+
- **Suitability**: 80% Yes, 20% No

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- Additional crop support
- More regional data
- UI/UX improvements
- Performance optimization
- Documentation
- Bug fixes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check documentation

---

## 🙏 Acknowledgments

- Gujarat Agricultural Department for data
- Indian Council of Agricultural Research (ICAR)
- Open-source community
- All contributors

---

## 📈 Future Roadmap

### Short Term
- [ ] Mobile app development
- [ ] Real-time weather integration
- [ ] Soil testing lab integration
- [ ] Market price predictions

### Long Term
- [ ] Satellite imagery analysis
- [ ] IoT sensor integration
- [ ] Blockchain for supply chain
- [ ] AI chatbot for queries
- [ ] Expansion to other states

---

**Version**: 2.0 (Improved)  
**Last Updated**: November 28, 2025  
**Maintained By**: Gujarat Crop Recommendation System Team
