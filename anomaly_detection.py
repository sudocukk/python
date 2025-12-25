"""
Anomaly Detection Module
This module detects abnormal values in happiness data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def detect_anomalies_iqr(df, column, factor=1.5):
    """
    Anomaly detection using IQR (Interquartile Range) method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
    anomalies['Anomaly_Type'] = np.where(anomalies[column] < lower_bound, 'Low', 'High')
    anomalies['Anomaly_Method'] = 'IQR'
    anomalies['IQR_Lower'] = lower_bound
    anomalies['IQR_Upper'] = upper_bound
    
    return anomalies


def detect_anomalies_zscore(df, column, threshold=3):
    """
    Anomaly detection using Z-Score method
    """
    z_scores = np.abs(stats.zscore(df[column]))
    anomalies = df[z_scores > threshold].copy()
    anomalies['Z_Score'] = z_scores[z_scores > threshold]
    anomalies['Anomaly_Method'] = 'Z-Score'
    anomalies['Threshold'] = threshold
    
    return anomalies


def detect_anomalies_isolation_forest(df, columns, contamination=0.1):
    """
    Anomaly detection using Isolation Forest method (multivariate)
    """
    # Prepare data
    data = df[columns].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(data_scaled)
    
    # Mark anomalies
    anomalies = df[predictions == -1].copy()
    anomalies['Anomaly_Score'] = iso_forest.score_samples(data_scaled[predictions == -1])
    anomalies['Anomaly_Method'] = 'Isolation Forest'
    anomalies['Contamination'] = contamination
    
    return anomalies, iso_forest


def comprehensive_anomaly_detection(df, columns_to_check=None):
    """
    Comprehensive anomaly detection - combines all methods
    """
    if columns_to_check is None:
        columns_to_check = ['Happiness Score', 'GDP per capita', 'Social support', 
                           'Healthy life expectancy', 'Freedom to make life choices']
    
    results = {
        'iqr_anomalies': {},
        'zscore_anomalies': {},
        'isolation_forest_anomalies': None,
        'summary': {}
    }
    
    print("="*70)
    print("ANOMALY DETECTION REPORT")
    print("="*70)
    
    # Univariate analysis with IQR method
    print("\n1. ANOMALY DETECTION WITH IQR METHOD:")
    print("-"*70)
    all_iqr_anomalies = []
    for col in columns_to_check:
        if col in df.columns:
            anomalies = detect_anomalies_iqr(df, col, factor=1.5)
            results['iqr_anomalies'][col] = anomalies
            all_iqr_anomalies.extend(anomalies.index.tolist())
            print(f"\n  {col}:")
            print(f"    - Number of anomalies detected: {len(anomalies)}")
            if len(anomalies) > 0:
                print(f"    - Low values: {len(anomalies[anomalies['Anomaly_Type'] == 'Low'])}")
                print(f"    - High values: {len(anomalies[anomalies['Anomaly_Type'] == 'High'])}")
    
    # Z-Score method
    print("\n2. ANOMALY DETECTION WITH Z-SCORE METHOD:")
    print("-"*70)
    all_zscore_anomalies = []
    for col in columns_to_check:
        if col in df.columns:
            anomalies = detect_anomalies_zscore(df, col, threshold=3)
            results['zscore_anomalies'][col] = anomalies
            all_zscore_anomalies.extend(anomalies.index.tolist())
            print(f"\n  {col}:")
            print(f"    - Number of anomalies detected: {len(anomalies)}")
            if len(anomalies) > 0:
                print(f"    - Average Z-Score: {anomalies['Z_Score'].mean():.2f}")
    
    # Isolation Forest (multivariate)
    print("\n3. ANOMALY DETECTION WITH ISOLATION FOREST METHOD:")
    print("-"*70)
    available_cols = [col for col in columns_to_check if col in df.columns]
    if len(available_cols) >= 2:
        anomalies_iso, model = detect_anomalies_isolation_forest(df, available_cols, contamination=0.1)
        results['isolation_forest_anomalies'] = anomalies_iso
        print(f"  - Number of anomalies detected: {len(anomalies_iso)}")
        print(f"  - Variables analyzed: {available_cols}")
        if len(anomalies_iso) > 0:
            print(f"  - Average anomaly score: {anomalies_iso['Anomaly_Score'].mean():.2f}")
    
    # Summary statistics
    print("\n4. SUMMARY STATISTICS:")
    print("-"*70)
    unique_iqr = len(set(all_iqr_anomalies))
    unique_zscore = len(set(all_zscore_anomalies))
    unique_iso = len(results['isolation_forest_anomalies']) if results['isolation_forest_anomalies'] is not None else 0
    
    results['summary'] = {
        'total_iqr_anomalies': unique_iqr,
        'total_zscore_anomalies': unique_zscore,
        'total_isolation_forest_anomalies': unique_iso,
        'percentage_iqr': (unique_iqr / len(df)) * 100,
        'percentage_zscore': (unique_zscore / len(df)) * 100,
        'percentage_iso': (unique_iso / len(df)) * 100 if unique_iso > 0 else 0
    }
    
    print(f"  - Unique anomalies detected with IQR: {unique_iqr} ({results['summary']['percentage_iqr']:.2f}%)")
    print(f"  - Unique anomalies detected with Z-Score: {unique_zscore} ({results['summary']['percentage_zscore']:.2f}%)")
    print(f"  - Anomalies detected with Isolation Forest: {unique_iso} ({results['summary']['percentage_iso']:.2f}%)")
    
    # Countries marked as anomalies
    all_anomaly_indices = set(all_iqr_anomalies + all_zscore_anomalies)
    if results['isolation_forest_anomalies'] is not None:
        all_anomaly_indices.update(results['isolation_forest_anomalies'].index.tolist())
    
    anomaly_countries = df.loc[list(all_anomaly_indices), 'Country name'].tolist() if 'Country name' in df.columns else []
    
    print(f"\n  - Total unique anomalies: {len(all_anomaly_indices)}")
    if len(anomaly_countries) > 0 and len(anomaly_countries) <= 20:
        print(f"  - Countries with anomalies: {', '.join(anomaly_countries[:10])}")
        if len(anomaly_countries) > 10:
            print(f"    ... and {len(anomaly_countries) - 10} more")
    
    print("\n" + "="*70)
    
    return results


def visualize_anomalies(df, results, column='Happiness Score'):
    """
    Visualize anomalies
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Box plot with IQR anomalies
    ax1 = axes[0, 0]
    if column in results['iqr_anomalies']:
        anomalies = results['iqr_anomalies'][column]
        normal_data = df[~df.index.isin(anomalies.index)]
        
        ax1.scatter(normal_data.index, normal_data[column], alpha=0.5, label='Normal', color='blue')
        ax1.scatter(anomalies.index, anomalies[column], alpha=0.8, label='Anomaly', color='red', s=100)
        ax1.boxplot([df[column]], positions=[len(df)//2], widths=len(df)//4, patch_artist=True)
        ax1.set_xlabel('Country Index')
        ax1.set_ylabel(column)
        ax1.set_title(f'IQR Method - {column}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Z-Score visualization
    ax2 = axes[0, 1]
    if column in df.columns:
        z_scores = np.abs(stats.zscore(df[column]))
        ax2.scatter(df.index, z_scores, alpha=0.6, c=z_scores, cmap='RdYlGn_r')
        ax2.axhline(y=3, color='r', linestyle='--', label='Threshold (3)')
        ax2.set_xlabel('Country Index')
        ax2.set_ylabel('Z-Score (Absolute Value)')
        ax2.set_title(f'Z-Score Analysis - {column}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Histogram with distribution
    ax3 = axes[1, 0]
    if column in df.columns:
        ax3.hist(df[column], bins=20, alpha=0.7, edgecolor='black')
        if column in results['iqr_anomalies']:
            anomalies = results['iqr_anomalies'][column]
            if len(anomalies) > 0:
                ax3.scatter(anomalies[column], [0]*len(anomalies), 
                          color='red', s=100, label='Anomalies', zorder=5)
        ax3.set_xlabel(column)
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Distribution and Anomalies - {column}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Isolation Forest results
    ax4 = axes[1, 1]
    if results['isolation_forest_anomalies'] is not None:
        iso_anomalies = results['isolation_forest_anomalies']
        normal_idx = df.index[~df.index.isin(iso_anomalies.index)]
        
        if column in df.columns:
            ax4.scatter(df.loc[normal_idx, column].index if len(normal_idx) > 0 else [], 
                       df.loc[normal_idx, column] if len(normal_idx) > 0 else [], 
                       alpha=0.5, label='Normal', color='blue')
            ax4.scatter(iso_anomalies.index, iso_anomalies[column], 
                       alpha=0.8, label='Anomaly', color='red', s=100)
            ax4.set_xlabel('Country Index')
            ax4.set_ylabel(column)
            ax4.set_title(f'Isolation Forest - {column}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'Country name': [f'Country_{i}' for i in range(100)],
        'Happiness Score': np.random.normal(5.5, 1.2, 100),
        'GDP per capita': np.exp(np.random.normal(9, 1, 100))
    })
    
    # Add some anomalies
    test_data.loc[10, 'Happiness Score'] = 9.5  # High anomaly
    test_data.loc[20, 'Happiness Score'] = 2.0   # Low anomaly
    test_data.loc[30, 'GDP per capita'] = 200000  # High anomaly
    
    results = comprehensive_anomaly_detection(test_data)
    print("\nTest completed!")
