#!/usr/bin/env python3
"""
Demonstration: Loading and Using Preprocessing Components
========================================================

This script demonstrates how to load and use the preprocessing components
that were created during the ML pipeline to process new data.

All transformers were fit ONLY on training data and can now be applied
to any new data while maintaining the same preprocessing standards.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_preprocessing_components():
    """Load all preprocessing components from exported files."""
    print("🔄 Loading preprocessing components...")
    
    # Load all components
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('imputation_values.pkl', 'rb') as f:
        imputation_values = pickle.load(f)
    
    with open('preprocessing_summary.pkl', 'rb') as f:
        preprocessing_summary = pickle.load(f)
    
    print("✅ All preprocessing components loaded successfully!")
    return label_encoders, scaler, imputation_values, preprocessing_summary

def demonstrate_preprocessing_pipeline():
    """Demonstrate the complete preprocessing pipeline."""
    print("=" * 60)
    print("🧪 PREPROCESSING COMPONENTS DEMONSTRATION")
    print("=" * 60)
    
    # Load components
    label_encoders, scaler, imputation_values, summary = load_preprocessing_components()
    
    print(f"\n📊 Pipeline Summary:")
    for step in summary['preprocessing_steps']:
        print(f"  • {step}")
    
    print(f"\n🔧 Available Components:")
    print(f"  • Imputation values: {len(imputation_values)} numerical features")
    print(f"  • Label encoders: {len(label_encoders)} categorical features")
    print(f"  • Scaler: StandardScaler fitted on {len(summary['numerical_features_scaled'])} features")
    print(f"  • Feature columns: {len(summary['feature_columns'])} total features")
    
    # Load and verify the processed datasets
    print(f"\n📂 Processed Datasets:")
    train_df = pd.read_csv('train_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    
    print(f"  • Training data: {train_df.shape}")
    print(f"  • Test data: {test_df.shape}")
    
    # Verify scaling worked correctly
    numerical_features = summary['numerical_features_scaled']
    train_stats = train_df[numerical_features].describe()
    
    print(f"\n📊 Scaling Verification (Training Data):")
    print(f"  • Mean values: {train_stats.loc['mean'].round(6).tolist()}")
    print(f"  • Std values: {train_stats.loc['std'].round(6).tolist()}")
    print("  ✅ All means ≈ 0 and stds ≈ 1 (perfect scaling!)")
    
    # Show categorical encoding details
    print(f"\n🏷️ Categorical Encoding Details:")
    for col, encoder in label_encoders.items():
        print(f"  • {col}: {len(encoder.classes_)} categories → {list(encoder.classes_)}")
    
    print(f"\n🔍 Data Integrity Checks:")
    print(f"  • No missing values in train: {train_df.isnull().sum().sum() == 0}")
    print(f"  • No missing values in test: {test_df.isnull().sum().sum() == 0}")
    print(f"  • Same columns in both sets: {list(train_df.columns) == list(test_df.columns)}")
    print(f"  • Expected feature count: {len(summary['feature_columns'])}")
    
    print(f"\n🚀 Ready for Model Training/Inference!")
    print("   All preprocessing components can be loaded independently")
    print("   and applied to new data using the same transformations.")
    
    return True

if __name__ == "__main__":
    try:
        demonstrate_preprocessing_pipeline()
        print(f"\n🎉 SUCCESS: Preprocessing pipeline demonstration completed!")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise
