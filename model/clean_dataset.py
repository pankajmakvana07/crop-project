"""
Dataset Cleaning Script - Fix unrealistic yields, pH values, and data quality issues
"""
import pandas as pd
import numpy as np

print("="*80)
print("DATASET CLEANING AND VALIDATION")
print("="*80)

# Load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'gujarat_full_crop_dataset.csv')
df = pd.read_csv(dataset_path, encoding='utf-8')
print(f"\nOriginal Dataset Shape: {df.shape}")

# ============================================================================
# STEP 1: Define Realistic Yield Ranges (Quintal per Hectare)
# ============================================================================
print("\n[STEP 1] Defining Realistic Yield Ranges...")

# Based on Gujarat agricultural data and Indian crop statistics
realistic_yield_ranges = {
    'Rice': (15, 40),
    'Wheat': (20, 35),
    'Cotton': (10, 25),
    'Bajra': (10, 25),
    'Groundnut': (15, 30),
    'Maize': (20, 45),
    'Jowar': (10, 25),
    'Castor': (8, 20),
    'Tur (Pigeon Pea)': (8, 15),
    'Moong (Green Gram)': (4, 12),
    'Urad (Black Gram)': (4, 12),
    'Sesame': (3, 10),
    'Sugarcane': (600, 1000),
    'Potato': (150, 300),
    'Onion': (150, 350),
    'Cumin': (3, 8),
    'Tobacco': (15, 25),
    'Mustard': (8, 18),
    'Rajma (Kidney Bean)': (8, 15),
    'Chickpea': (10, 20),
    'Soybean': (10, 25)
}

# ============================================================================
# STEP 2: Fix Unrealistic Yields
# ============================================================================
print("\n[STEP 2] Fixing Unrealistic Yields...")

fixed_count = 0
for crop, (min_yield, max_yield) in realistic_yield_ranges.items():
    crop_mask = df['Crop_Recommended'] == crop
    
    # Find unrealistic yields
    too_low = (df['Crop_Yield_Quintal_per_Hectare'] < min_yield) & crop_mask
    too_high = (df['Crop_Yield_Quintal_per_Hectare'] > max_yield) & crop_mask
    
    if too_low.sum() > 0:
        # Replace with random value in lower range
        df.loc[too_low, 'Crop_Yield_Quintal_per_Hectare'] = np.random.uniform(
            min_yield, min_yield + (max_yield - min_yield) * 0.3, too_low.sum()
        )
        fixed_count += too_low.sum()
    
    if too_high.sum() > 0:
        # Replace with random value in upper range
        df.loc[too_high, 'Crop_Yield_Quintal_per_Hectare'] = np.random.uniform(
            max_yield - (max_yield - min_yield) * 0.3, max_yield, too_high.sum()
        )
        fixed_count += too_high.sum()

print(f"✓ Fixed {fixed_count} unrealistic yield values")

# ============================================================================
# STEP 3: Fix Soil pH Values
# ============================================================================
print("\n[STEP 3] Fixing Soil pH Values...")

# Realistic pH range: 4.5 to 9.0 (most crops: 5.5 to 8.5)
ph_fixed = 0

# Fix extreme pH values
extreme_low_ph = df['Soil_pH'] < 4.5
extreme_high_ph = df['Soil_pH'] > 9.0

if extreme_low_ph.sum() > 0:
    df.loc[extreme_low_ph, 'Soil_pH'] = np.random.uniform(5.5, 6.5, extreme_low_ph.sum())
    ph_fixed += extreme_low_ph.sum()

if extreme_high_ph.sum() > 0:
    df.loc[extreme_high_ph, 'Soil_pH'] = np.random.uniform(7.5, 8.5, extreme_high_ph.sum())
    ph_fixed += extreme_high_ph.sum()

print(f"✓ Fixed {ph_fixed} extreme pH values")
print(f"  pH Range: {df['Soil_pH'].min():.2f} to {df['Soil_pH'].max():.2f}")

# ============================================================================
# STEP 4: Remove Duplicates
# ============================================================================
print("\n[STEP 4] Removing Duplicates...")

# Check for exact duplicates
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
print(f"✓ Removed {duplicates_before} exact duplicate rows")

# Check for near-duplicates (same location, month, year, crop)
key_cols = ['District_Name', 'Taluka_Name', 'Month', 'Year', 'Crop_Recommended']
near_duplicates_before = df.duplicated(subset=key_cols).sum()
df = df.drop_duplicates(subset=key_cols, keep='first')
print(f"✓ Removed {near_duplicates_before} near-duplicate rows")

print(f"  Dataset shape after deduplication: {df.shape}")

# ============================================================================
# STEP 5: Balance Suitability Classes
# ============================================================================
print("\n[STEP 5] Balancing Suitability Classes...")

suit_dist_before = df['Crop_Suitability'].value_counts()
print(f"Before balancing: {suit_dist_before.to_dict()}")

# If heavily imbalanced, create more "No" examples
yes_count = (df['Crop_Suitability'] == 'Yes').sum()
no_count = (df['Crop_Suitability'] == 'No').sum()

target_no_ratio = 0.20  # Aim for 20% "No" examples

if no_count / len(df) < target_no_ratio:
    needed_no = int(len(df) * target_no_ratio) - no_count
    
    # Select records with poor conditions to mark as "No"
    yes_records = df[df['Crop_Suitability'] == 'Yes'].copy()
    
    # Score records based on how unsuitable they are
    yes_records['unsuitability_score'] = 0
    
    # Low yield indicator
    for crop in yes_records['Crop_Recommended'].unique():
        if crop in realistic_yield_ranges:
            min_yield, max_yield = realistic_yield_ranges[crop]
            crop_mask = yes_records['Crop_Recommended'] == crop
            # Normalize yield to 0-1 scale, then invert (low yield = high unsuitability)
            yes_records.loc[crop_mask, 'unsuitability_score'] += (
                1 - (yes_records.loc[crop_mask, 'Crop_Yield_Quintal_per_Hectare'] - min_yield) / (max_yield - min_yield)
            )
    
    # Extreme pH indicator
    yes_records['unsuitability_score'] += (
        (yes_records['Soil_pH'] < 5.8) | (yes_records['Soil_pH'] > 8.2)
    ).astype(float) * 0.5
    
    # Select top unsuitable records
    unsuitable_indices = yes_records.nlargest(min(needed_no, len(yes_records)), 'unsuitability_score').index
    
    # Mark as "No" and reduce yield
    df.loc[unsuitable_indices, 'Crop_Suitability'] = 'No'
    df.loc[unsuitable_indices, 'Crop_Yield_Quintal_per_Hectare'] *= 0.5
    
    print(f"✓ Created {len(unsuitable_indices)} additional 'No' examples")

suit_dist_after = df['Crop_Suitability'].value_counts()
print(f"After balancing: {suit_dist_after.to_dict()}")

# ============================================================================
# STEP 6: Validate and Fix NPK Values
# ============================================================================
print("\n[STEP 6] Validating NPK Values...")

# Realistic NPK ranges (kg/ha)
npk_ranges = {
    'Nitrogen': (50, 500),
    'Phosphorus': (5, 100),
    'Potassium': (50, 700)
}

npk_fixed = 0
for nutrient, (min_val, max_val) in npk_ranges.items():
    too_low = df[nutrient] < min_val
    too_high = df[nutrient] > max_val
    
    if too_low.sum() > 0:
        df.loc[too_low, nutrient] = np.random.uniform(min_val, min_val * 1.5, too_low.sum())
        npk_fixed += too_low.sum()
    
    if too_high.sum() > 0:
        df.loc[too_high, nutrient] = np.random.uniform(max_val * 0.7, max_val, too_high.sum())
        npk_fixed += too_high.sum()

print(f"✓ Fixed {npk_fixed} NPK values")

# ============================================================================
# STEP 7: Add Missing Rainfall/Season Data
# ============================================================================
print("\n[STEP 7] Validating Rainfall and Season Data...")

# Fix missing or zero rainfall in monsoon months
monsoon_months = [6, 7, 8, 9]
monsoon_mask = df['Month'].isin(monsoon_months)
low_monsoon_rain = monsoon_mask & (df['Rainfall_mm'] < 50)

if low_monsoon_rain.sum() > 0:
    # Add realistic monsoon rainfall
    df.loc[low_monsoon_rain, 'Rainfall_mm'] = np.random.uniform(100, 400, low_monsoon_rain.sum())
    print(f"✓ Fixed {low_monsoon_rain.sum()} low monsoon rainfall values")

# Ensure winter months have low rainfall
winter_months = [11, 12, 1, 2]
winter_mask = df['Month'].isin(winter_months)
high_winter_rain = winter_mask & (df['Rainfall_mm'] > 50)

if high_winter_rain.sum() > 0:
    df.loc[high_winter_rain, 'Rainfall_mm'] = np.random.uniform(0, 30, high_winter_rain.sum())
    print(f"✓ Fixed {high_winter_rain.sum()} high winter rainfall values")

# ============================================================================
# STEP 8: Final Validation
# ============================================================================
print("\n[STEP 8] Final Validation...")

print("\nDataset Statistics:")
print(f"  Total Records: {len(df)}")
print(f"  Unique Crops: {df['Crop_Recommended'].nunique()}")
print(f"  Suitability Distribution: {df['Crop_Suitability'].value_counts().to_dict()}")
print(f"  Yield Range: {df['Crop_Yield_Quintal_per_Hectare'].min():.2f} to {df['Crop_Yield_Quintal_per_Hectare'].max():.2f}")
print(f"  pH Range: {df['Soil_pH'].min():.2f} to {df['Soil_pH'].max():.2f}")

# Check for any remaining issues
print("\nData Quality Checks:")
print(f"  Missing Values: {df.isnull().sum().sum()}")
print(f"  Negative Yields: {(df['Crop_Yield_Quintal_per_Hectare'] < 0).sum()}")
print(f"  Extreme pH (<4.5 or >9.0): {((df['Soil_pH'] < 4.5) | (df['Soil_pH'] > 9.0)).sum()}")

# ============================================================================
# STEP 9: Save Cleaned Dataset
# ============================================================================
print("\n[STEP 9] Saving Cleaned Dataset...")

# Backup original if not already backed up
backup_path = os.path.join(script_dir, 'gujarat_full_crop_dataset_original.csv')
if not os.path.exists(backup_path):
    df_original = pd.read_csv(dataset_path, encoding='utf-8')
    df_original.to_csv(backup_path, index=False, encoding='utf-8')
    print("✓ Original dataset backed up to: gujarat_full_crop_dataset_original.csv")

# Save cleaned dataset
df.to_csv(dataset_path, index=False, encoding='utf-8')
print("✓ Cleaned dataset saved to: gujarat_full_crop_dataset.csv")

print("\n" + "="*80)
print("✅ DATASET CLEANING COMPLETE!")
print("="*80)
print("\nNext step: Run train_models.py to train with cleaned data")
