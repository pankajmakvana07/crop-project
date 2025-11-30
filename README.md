# 🌾 Gujarat Crop Recommendation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-336791.svg)](https://www.postgresql.org/)

An AI-powered web application that provides intelligent crop recommendations for farmers in Gujarat, India, based on soil conditions, climate data, and agricultural best practices.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Gujarat+Crop+Recommendation+System)

---

## 🎯 Quick Start

### Prerequisites
- Python 3.8 or higher
- PostgreSQL 12 or higher
- pip package manager

### Installation (5 minutes)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```env
   DB_HOST=localhost
   DB_NAME=crop_db
   DB_USER=postgres
   DB_PASS=your_password
   DB_PORT=5432
   sender_mail=your_email@gmail.com
   sender_mail_pass=your_app_password
   AUTH_SECRET=your_secret_key
   ```

5. **Initialize database**
   ```bash
   python db/create_db.py
   ```

6. **Run the application**
   ```bash
   streamlit run navigation.py
   ```

7. **Open your browser**
   ```
   http://localhost:8501
   ```

---

## ✨ Features

### 🌱 Core Functionality
- **Smart Crop Recommendations**: Get top 3 crop suggestions based on your soil
- **Suitability Analysis**: Check if specific crops can grow in your region
- **Yield Predictions**: Estimate expected harvest in quintal/hectare
- **Treatment Plans**: Detailed fertilizer and pesticide recommendations

### 🔐 User Management
- Secure authentication with bcrypt encryption
- Email and phone number login
- OTP verification support
- Guest mode for quick access

### 🌍 Regional Support
- All Gujarat districts and talukas
- Soil type classification (Sandy, Clay, Loamy, etc.)
- pH level tracking and analysis
- Climate-aware recommendations

### 🌐 Multi-language Support
- Translation to 12 Indian languages
- Hindi, Gujarati, Marathi, Tamil, Telugu, Kannada, Malayalam, Punjabi, Bengali, Urdu, Odia, Assamese

### 📊 Analytics & History
- Track all predictions
- View historical recommendations
- Statistics dashboard
- Export capabilities

---

## 🤖 Machine Learning Models

### Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Crop Classification | Accuracy | 87.51% |
| Crop Classification | Top-3 Accuracy | 99.47% |
| Suitability | ROC AUC | 86.72% |
| Suitability | Precision | 89.80% |
| Yield Prediction | R² Score | 1.00 |

### Supported Crops (21)
Rice, Wheat, Cotton, Bajra, Groundnut, Maize, Jowar, Castor, Tur (Pigeon Pea), Moong (Green Gram), Urad (Black Gram), Sesame, Sugarcane, Potato, Onion, Cumin, Tobacco, Mustard, Rajma (Kidney Bean), Chickpea, Soybean

### Training Data
- **17,818 records** from Gujarat agricultural data
- **52 features** including soil, climate, and location
- **Domain knowledge integration** for accuracy
- **SMOTE balancing** for fair predictions

---

## 📁 Project Structure

```
project/
├── model/                  # ML models and training scripts
│   ├── improved_Amodel.py # Enhanced model training
│   ├── crop_recommendation_models.pkl
│   ├── crop_treatments.json
│   └── gujarat_full_crop_dataset.csv
├── page/                   # UI pages
│   ├── app.py             # Main dashboard
│   ├── login_page.py      # Authentication
│   ├── AddDetails.py      # Soil data management
│   └── pridect_dynamic.py # Prediction interface
├── utils/                  # Utility modules
│   ├── model_integration.py
│   ├── db_handler.py
│   └── auth_token.py
├── db/                     # Database scripts
│   └── create_db.py
├── navigation.py           # Main app router
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── README.md              # This file
├── PROJECT_OVERVIEW.md    # Detailed documentation
└── LICENSE                # MIT License
```

---

## 🚀 Usage

### For Farmers

1. **Sign Up / Login**
   - Create an account or use guest mode
   - Verify with OTP if needed

2. **Add Your Soil Details**
   - Navigate to "Add Soil Details"
   - Select your district and taluka
   - Choose soil type and enter pH level
   - Save your information

3. **Get Crop Recommendations**
   - Click "Crop Prediction"
   - Choose prediction mode:
     - **Normal**: Get top 3 recommended crops
     - **Advanced**: Check specific crop suitability
   - View results with treatment plans

4. **Review History**
   - Check past predictions
   - Track your farming decisions
   - Compare recommendations

### For Developers

#### Retrain Models
```bash
python model/improved_Amodel.py
```

#### Test Predictions
```bash
python model/test_improved_predictions.py
```

#### Reset Database
```bash
python db/delete_db.py
python db/create_db.py
```

#### Run Tests
```bash
python -m pytest tests/
```

---

## 🔧 Configuration

### Database Setup

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # Windows: Download from postgresql.org
   # Mac: brew install postgresql
   ```

2. **Create Database**
   ```sql
   CREATE DATABASE crop_db;
   CREATE USER crop_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE crop_db TO crop_user;
   ```

3. **Update .env**
   ```env
   DB_HOST=localhost
   DB_NAME=crop_db
   DB_USER=crop_user
   DB_PASS=your_password
   DB_PORT=5432
   ```

### Email Configuration (Optional)

For OTP and notifications:

1. **Enable 2FA on Gmail**
2. **Generate App Password**
3. **Update .env**
   ```env
   sender_mail=your_email@gmail.com
   sender_mail_pass=your_16_char_app_password
   ```

---

## 📊 API Usage

### Python Integration

```python
from utils.model_integration import get_prediction_service

# Initialize service
service = get_prediction_service()

# Normal prediction (top 3 crops)
result = service.predict_normal(
    user_id=1,
    top_k=3
)

# Advanced prediction (specific crop)
result = service.predict_advanced(
    user_id=1,
    crop_name="Rice"
)

# Get available crops
crops = service.get_available_crops()
```

### Response Format

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
      "treatment": {
        "pestManagement": "IPM for stem borers",
        "recommendedIrrigation": "Flooded irrigation",
        "recommendedFertilizers": ["Urea", "DAP"],
        "recommendedPesticides": ["Chlorantraniliprole"]
      }
    }
  ]
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Database Connection Error**
```
Solution: Check PostgreSQL is running and credentials in .env are correct
```

**2. Model Files Not Found**
```bash
# Retrain models
python model/improved_Amodel.py
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. Port Already in Use**
```bash
# Change Streamlit port
streamlit run navigation.py --server.port 8502
```

**5. Translation Not Working**
```
Solution: Check internet connection (Google Translate API requires internet)
```

---

## 📚 Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: Comprehensive project documentation
- **[MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md)**: Model enhancement details
- **[LICENSE](LICENSE)**: MIT License terms

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Areas
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation
- 🌐 Translations
- 🎨 UI/UX improvements
- 🧪 Tests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Gujarat Crop Recommendation System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **Gujarat Agricultural Department** for providing agricultural data
- **Indian Council of Agricultural Research (ICAR)** for crop guidelines
- **Open-source community** for amazing tools and libraries
- **All contributors** who helped improve this project

---

## 📞 Support

### Get Help
- 📖 Read the [documentation](PROJECT_OVERVIEW.md)
- 🐛 Report [issues](https://github.com/your-repo/issues)
- 💬 Join discussions
- 📧 Contact the team

### Resources
- [Streamlit Documentation](https://docs.streamlit.io)
- [PostgreSQL Guide](https://www.postgresql.org/docs/)
- [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 🗺️ Roadmap

### Version 2.1 (Q1 2026)
- [ ] Mobile app (Android/iOS)
- [ ] Real-time weather integration
- [ ] Soil testing lab integration
- [ ] Market price predictions

### Version 3.0 (Q3 2026)
- [ ] Satellite imagery analysis
- [ ] IoT sensor integration
- [ ] Blockchain for supply chain
- [ ] AI chatbot for queries

### Future
- [ ] Expansion to other Indian states
- [ ] Drone-based crop monitoring
- [ ] Automated irrigation recommendations
- [ ] Disease detection system

---

## 📈 Statistics

- **17,818** training data points
- **21** supported crops
- **250+** talukas covered
- **87.51%** prediction accuracy
- **99.47%** top-3 accuracy
- **12** supported languages

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

**Made with ❤️ for Gujarat Farmers**

**Version**: 2.0 (Improved)  
**Last Updated**: November 28, 2025  
**Status**: Active Development

---

## 📸 Screenshots

### Dashboard
![Dashboard](https://via.placeholder.com/800x400?text=Dashboard+View)

### Crop Prediction
![Prediction](https://via.placeholder.com/800x400?text=Prediction+Interface)

### Treatment Plans
![Treatment](https://via.placeholder.com/800x400?text=Treatment+Recommendations)

---

**Happy Farming! 🌾**
