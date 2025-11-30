#!/usr/bin/env python3
"""
Dataset Quality Check and Improvement Script
Validates the improvements made to the crop prediction model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATASET QUALITY CHECK AND VALIDATION")
print("="*80)

# Load the dataset
print("\n[STEP 1] Loading dataset...")
import os
dataset_path = 'gujarat_full_crop_dataset.csv'
if not os.path.exists(dataset_path):
    dataset_path = os.path.join('model', 'gujarat_full_crop_dataset.csv')
df = pd.read_csv(dataset_path, encoding='utf-8')
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

# ============================================================================
# STEP 1: DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*60)
print("[STEP 1] DATA QUALITY ASSESSMENT")
print("="*60)

# Check for missing values
print("\n1.1 Missing Values Analysis:")
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_summary.index,
    'Missing_Count': missing_summary.values,
    'Missing_Percentage': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print("Missing values found:")
    print(missing_df.to_string(index=False))
else:
    print("✓ No missing values found!")

# Check for duplicates
print(f"\n1.2 Duplicate Records:")
duplicates = df.duplicated().sum()
print(f"Exact duplicates: {duplicates}")

# Check data types
print(f"\n1.3 Data Types:")
print(df.dtypes.value_counts())

# ============================================================================
# STEP 2: CROP-SOIL-pH COMPATIBILITY ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("[STEP 2] CROP-SOIL-pH COMPATIBILITY ANALYSIS")
print("="*60)

def check_crop_soil_compatibility(crop, soil_type, ph):
    """Check if crop-soil-pH combination is realistic"""
    
    # Define ideal conditions for major crops
    crop_requirements = {
        'Tobacco': {
            'ideal_soils': ['Sandy', 'Sandy Loam', 'Loamy'],
            'acceptable_soils': ['Black Cotton', 'Clay'],
            'ideal_ph': (6.0, 7.5),
            'acceptable_ph': (5.8, 7.8)
        },
        'Rice': {
            'ideal_soils': ['Clay', 'Loamy', 'Black Cotton', 'Silty'],
            'acceptable_soils': ['Sandy Loam'],
            'ideal_ph': (5.5, 7.0),
            'acceptable_ph': (5.0, 7.5)
        },
        'Wheat': {
            'ideal_soils': ['Loamy', 'Clay', 'Black Cotton', 'Sandy Loam'],
            'acceptable_soils': ['Silty'],
            'ideal_ph': (6.0, 7.5),
            'acceptable_ph': (5.5, 8.0)
        },
        'Cotton': {
            'ideal_soils': ['Black Cotton', 'Clay', 'Loamy'],
            'acceptable_soils': ['Silty'],
            'ideal_ph': (6.5, 8.0),
            'acceptable_ph': (6.0, 8.5)
        },
        'Groundnut': {
            'ideal_soils': ['Sandy', 'Sandy Loam', 'Loamy'],
            'acceptable_soils': [],
            'ideal_ph': (6.0, 7.0),
            'acceptable_ph': (5.5, 7.5)
        },
        'Bajra': {
            'ideal_soils': ['Sandy', 'Sandy Loam', 'Loamy'],
            'acceptable_soils': ['Black Cotton', 'Clay', 'Silty'],
            'ideal_ph': (6.5, 7.5),
            'acceptable_ph': (6.0, 8.0)
        }
    }
    
    if crop not in crop_requirements:
        return 'unknown', 0.5  # Default for unknown crops
    
    req = crop_requirements[crop]
    
    # Check soil compatibility
    if soil_type in req['ideal_soils']:
        soil_score = 1.0
    elif soil_type in req['acceptable_soils']:
        soil_score = 0.7
    else:
        soil_score = 0.3
    
    # Check pH compatibility
    if req['ideal_ph'][0] <= ph <= req['ideal_ph'][1]:
        ph_score = 1.0
    elif req['acceptable_ph'][0] <= ph <= req['acceptable_ph'][1]:
        ph_score = 0.7
    else:
        ph_score = 0.3
    
    # Combined score
    combined_score = (soil_score + ph_score) / 2
    
    if combined_score >= 0.8:
        return 'ideal', combined_score
    elif combined_score >= 0.6:
        return 'good', combined_score
    elif combined_score >= 0.4:
        return 'acceptable', combined_score
    else:
        return 'poor', combined_score

# Analyze crop-soil-pH compatibility
print("\n2.1 Analyzing crop-soil-pH compatibility...")
compatibility_results = []

for idx, row in df.iterrows():
    crop = row['Crop_Recommended']
    soil = row['Soil_Type']
    ph = row['Soil_pH']
    suitability = row['Crop_Suitability']
    
    compatibility, score = check_crop_soil_compatibility(crop, soil, ph)
    
    compatibility_results.append({
        'Index': idx,
        'Crop': crop,
        'Soil_Type': soil,
        'pH': ph,
        'Labeled_Suitability': suitability,
        'Compatibility': compatibility,
        'Score': score
    })

compatibility_df = pd.DataFrame(compatibility_results)

# Summary of compatibility analysis
print("\n2.2 Compatibility Summary:")
compatibility_summary = compatibility_df['Compatibility'].value_counts()
print(compatibility_summary)

# Identify potential mislabeled records
print("\n2.3 Potential Mislabeling Analysis:")

# Records labeled as "Yes" but have poor compatibility
poor_but_yes = compatibility_df[
    (compatibility_df['Compatibility'] == 'poor') & 
    (compatibility_df['Labeled_Suitability'] == 'Yes')
]

# Records labeled as "No" but have good/ideal compatibility
good_but_no = compatibility_df[
    (compatibility_df['Compatibility'].isin(['ideal', 'good'])) & 
    (compatibility_df['Labeled_Suitability'] == 'No')
]

print(f"Records labeled 'Yes' but have poor compatibility: {len(poor_but_yes)}")
print(f"Records labeled 'No' but have good/ideal compatibility: {len(good_but_no)}")

if len(poor_but_yes) > 0:
    print("\nTop 10 'Yes' records with poor compatibility:")
    print(poor_but_yes[['Crop', 'Soil_Type', 'pH', 'Score']].head(10).to_string(index=False))

if len(good_but_no) > 0:
    print("\nTop 10 'No' records with good compatibility:")
    print(good_but_no[['Crop', 'Soil_Type', 'pH', 'Score']].head(10).to_string(index=False))

# ============================================================================
# STEP 3: TOBACCO-ANAND SPECIFIC ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("[STEP 3] TOBACCO-ANAND SPECIFIC ANALYSIS")
print("="*60)

# Check for Tobacco records in Anand district
anand_records = df[df['District_Name'].str.contains('ANAND', na=False, case=False)]
tobacco_records = df[df['Crop_Recommended'] == 'Tobacco']
anand_tobacco = df[
    (df['District_Name'].str.contains('ANAND', na=False, case=False)) & 
    (df['Crop_Recommended'] == 'Tobacco')
]

print(f"\n3.1 Record Counts:")
print(f"Total records in Anand district: {len(anand_records)}")
print(f"Total Tobacco records: {len(tobacco_records)}")
print(f"Tobacco records in Anand: {len(anand_tobacco)}")

if len(anand_tobacco) > 0:
    print(f"\n3.2 Anand-Tobacco Records Analysis:")
    print("Soil types in Anand-Tobacco records:")
    print(anand_tobacco['Soil_Type'].value_counts())
    
    print("\nSuitability labels in Anand-Tobacco records:")
    print(anand_tobacco['Crop_Suitability'].value_counts())
    
    print("\npH range in Anand-Tobacco records:")
    print(f"Min pH: {anand_tobacco['Soil_pH'].min():.2f}")
    print(f"Max pH: {anand_tobacco['Soil_pH'].max():.2f}")
    print(f"Mean pH: {anand_tobacco['Soil_pH'].mean():.2f}")
    
    # Check if these records are now correctly labeled
    anand_tobacco_suitable = anand_tobacco[anand_tobacco['Crop_Suitability'] == 'Yes']
    print(f"\nCorrectly labeled as suitable: {len(anand_tobacco_suitable)}/{len(anand_tobacco)}")
    
    if len(anand_tobacco_suitable) > 0:
        print("✓ FIXED: Tobacco records in Anand are now correctly labeled as suitable!")
    else:
        print("⚠ ISSUE: Tobacco records in Anand are still labeled as unsuitable")
else:
    print("⚠ No Tobacco records found in Anand district")

# ============================================================================
# STEP 4: YIELD OUTLIER ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("[STEP 4] YIELD OUTLIER ANALYSIS")
print("="*60)

print("\n4.1 Yield Statistics by Crop (Top 10 crops):")
top_crops = df['Crop_Recommended'].value_counts().head(10).index

for crop in top_crops:
    crop_data = df[df['Crop_Recommended'] == crop]['Crop_Yield_Quintal_per_Hectare']
    print(f"\n{crop}:")
    print(f"  Count: {len(crop_data)}")
    print(f"  Mean: {crop_data.mean():.2f}")
    print(f"  Std: {crop_data.std():.2f}")
    print(f"  Min: {crop_data.min():.2f}")
    print(f"  Max: {crop_data.max():.2f}")
    print(f"  Q1: {crop_data.quantile(0.25):.2f}")
    print(f"  Q3: {crop_data.quantile(0.75):.2f}")

# Identify extreme outliers
print("\n4.2 Extreme Yield Outliers:")
extreme_outliers = []

for crop in df['Crop_Recommended'].unique():
    crop_data = df[df['Crop_Recommended'] == crop]
    if len(crop_data) < 5:  # Skip crops with too few samples
        continue
    
    yields = crop_data['Crop_Yield_Quintal_per_Hectare']
    Q1 = yields.quantile(0.25)
    Q3 = yields.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define extreme outliers (3 * IQR)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = crop_data[
        (crop_data['Crop_Yield_Quintal_per_Hectare'] < lower_bound) |
        (crop_data['Crop_Yield_Quintal_per_Hectare'] > upper_bound)
    ]
    
    if len(outliers) > 0:
        extreme_outliers.extend(outliers.index.tolist())

print(f"Total extreme yield outliers identified: {len(extreme_outliers)}")

# ============================================================================
# STEP 5: CLASS BALANCE ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("[STEP 5] CLASS BALANCE ANALYSIS")
print("="*60)

print("\n5.1 Crop Distribution:")
crop_counts = df['Crop_Recommended'].value_counts()
print(f"Total unique crops: {len(crop_counts)}")
print(f"Most common crop: {crop_counts.index[0]} ({crop_counts.iloc[0]} records)")
print(f"Least common crop: {crop_counts.index[-1]} ({crop_counts.iloc[-1]} records)")

# Calculate imbalance ratio
max_count = crop_counts.max()
min_count = crop_counts.min()
imbalance_ratio = max_count / min_count
print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")

# Identify severely under-represented crops
avg_count = crop_counts.mean()
underrepresented = crop_counts[crop_counts < avg_count * 0.1]
print(f"\nSeverely under-represented crops (< 10% of average):")
print(underrepresented)

print("\n5.2 Suitability Distribution:")
suitability_counts = df['Crop_Suitability'].value_counts()
print(suitability_counts)
suitability_ratio = suitability_counts['Yes'] / suitability_counts['No'] if 'No' in suitability_counts else float('inf')
print(f"Yes:No ratio: {suitability_ratio:.2f}:1")

# ============================================================================
# STEP 6: GENERATE IMPROVEMENT RECOMMENDATIONS
# ============================================================================
print("\n" + "="*60)
print("[STEP 6] IMPROVEMENT RECOMMENDATIONS")
print("="*60)

recommendations = []

# Data quality recommendations
if duplicates > 0:
    recommendations.append(f"Remove {duplicates} duplicate records")

if len(extreme_outliers) > 0:
    recommendations.append(f"Review and potentially remove {len(extreme_outliers)} extreme yield outliers")

# Labeling recommendations
if len(poor_but_yes) > 0:
    recommendations.append(f"Review {len(poor_but_yes)} records labeled 'Yes' with poor crop-soil compatibility")

if len(good_but_no) > 0:
    recommendations.append(f"Review {len(good_but_no)} records labeled 'No' with good crop-soil compatibility")

# Balance recommendations
if imbalance_ratio > 10:
    recommendations.append(f"Address severe class imbalance (ratio: {imbalance_ratio:.1f}:1)")

if len(underrepresented) > 0:
    recommendations.append(f"Add more samples for {len(underrepresented)} under-represented crops")

# Tobacco-Anand specific
if len(anand_tobacco) == 0:
    recommendations.append("Add Tobacco cultivation records for Anand district (known tobacco growing region)")

print("\nRecommendations for dataset improvement:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

if len(recommendations) == 0:
    print("✓ Dataset appears to be in good condition!")

# ============================================================================
# STEP 7: SAVE ANALYSIS RESULTS
# ============================================================================
print("\n" + "="*60)
print("[STEP 7] SAVING ANALYSIS RESULTS")
print("="*60)

# Save compatibility analysis
compatibility_df.to_csv('dataset_compatibility_analysis.csv', index=False)
print("✓ Saved: dataset_compatibility_analysis.csv")

# Save potential mislabeled records
if len(poor_but_yes) > 0 or len(good_but_no) > 0:
    mislabeled_df = pd.concat([poor_but_yes, good_but_no])
    mislabeled_df.to_csv('potential_mislabeled_records.csv', index=False)
    print("✓ Saved: potential_mislabeled_records.csv")

# Save extreme outliers
if len(extreme_outliers) > 0:
    outlier_df = df.loc[extreme_outliers]
    outlier_df.to_csv('extreme_yield_outliers.csv', index=False)
    print("✓ Saved: extreme_yield_outliers.csv")

# Create summary report
summary_report = f"""
DATASET QUALITY ANALYSIS SUMMARY
================================

Dataset Size: {df.shape[0]} records, {df.shape[1]} columns
Missing Values: {len(missing_df)} columns with missing data
Duplicate Records: {duplicates}
Unique Crops: {len(crop_counts)}
Class Imbalance Ratio: {imbalance_ratio:.2f}:1

Compatibility Analysis:
- Ideal conditions: {compatibility_summary.get('ideal', 0)} records
- Good conditions: {compatibility_summary.get('good', 0)} records
- Acceptable conditions: {compatibility_summary.get('acceptable', 0)} records
- Poor conditions: {compatibility_summary.get('poor', 0)} records

Potential Issues:
- Records labeled 'Yes' with poor compatibility: {len(poor_but_yes)}
- Records labeled 'No' with good compatibility: {len(good_but_no)}
- Extreme yield outliers: {len(extreme_outliers)}
- Under-represented crops: {len(underrepresented)}

Tobacco-Anand Analysis:
- Anand district records: {len(anand_records)}
- Tobacco records: {len(tobacco_records)}
- Anand-Tobacco records: {len(anand_tobacco)}

Recommendations: {len(recommendations)} improvement suggestions identified
"""

with open('dataset_quality_report.txt', 'w') as f:
    f.write(summary_report)
print("✓ Saved: dataset_quality_report.txt")

print(f"\n{'='*80}")
print("DATASET QUALITY CHECK COMPLETED")
print(f"{'='*80}")
print(f"Analysis complete! Check the generated files for detailed results.")