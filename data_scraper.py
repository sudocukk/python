"""
Web Scraping Module - Fetching World Happiness Report Data
This module fetches data from the web to obtain messy data and demonstrates the cleaning process.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np

def scrape_wikipedia_happiness_data():
    """
    Fetches World Happiness Report data from Wikipedia.
    Note: This produces messy data that needs to be cleaned.
    """
    print("Starting web scraping...")
    
    # Fetch table from Wikipedia
    url = "https://en.wikipedia.org/wiki/World_Happiness_Report"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find tables
        tables = soup.find_all('table', class_='wikitable')
        
        if tables:
            # Get first table (usually the most recent year's data)
            df = pd.read_html(str(tables[0]))[0]
            print(f"Fetched {len(df)} rows of data from Wikipedia.")
            return df
        else:
            print("Table not found. Creating sample data...")
            return create_messy_sample_data()
            
    except Exception as e:
        print(f"Web scraping error: {e}")
        print("Creating sample messy data...")
        return create_messy_sample_data()


def create_messy_sample_data():
    """
    Creates sample messy data - simulates real web scraping
    This data will be irregular like real data fetched from the web.
    """
    np.random.seed(42)
    
    countries = ['Finland', 'Denmark', 'Switzerland', 'Iceland', 'Netherlands', 
                'Norway', 'Sweden', 'Luxembourg', 'New Zealand', 'Austria',
                'Australia', 'Israel', 'Germany', 'Canada', 'Ireland',
                'Costa Rica', 'United Kingdom', 'Czech Republic', 'United States', 'Belgium',
                'France', 'Bahrain', 'Malta', 'Taiwan', 'United Arab Emirates',
                'Saudi Arabia', 'Spain', 'Italy', 'Slovenia', 'Guatemala',
                'Singapore', 'Romania', 'Poland', 'Kuwait', 'Serbia',
                'Chile', 'Bahamas', 'Argentina', 'Hungary', 'Trinidad and Tobago',
                'Panama', 'Nicaragua', 'Colombia', 'Estonia', 'Jamaica',
                'Mexico', 'Uruguay', 'Lithuania', 'Slovakia', 'Ecuador',
                'Japan', 'South Korea', 'Philippines', 'Brazil', 'Thailand',
                'Portugal', 'Latvia', 'South Africa', 'India', 'China',
                'Russia', 'Turkey', 'Greece', 'Bulgaria', 'Morocco',
                'Algeria', 'Tunisia', 'Egypt', 'Bangladesh', 'Pakistan',
                'Nigeria', 'Kenya', 'Tanzania', 'Zimbabwe', 'Rwanda',
                'Afghanistan', 'Central African Republic', 'South Sudan']
    
    n = len(countries)
    
    # Messy data characteristics:
    # 1. Mixed formats (string, number mixed)
    # 2. Missing values (NaN, "N/A", "", "-")
    # 3. Different decimal separators
    # 4. Extra spaces
    # 5. Special characters
    
    messy_data = []
    missing_patterns = [None, "N/A", "", "-", "n/a", "NULL"]
    
    for i, country in enumerate(countries):
        row = {}
        
        # Country name - sometimes with extra spaces
        if i % 5 == 0:
            row['Country'] = f"  {country}  "  # Extra spaces
        else:
            row['Country'] = country
        
        # Happiness score - mixed format
        happiness = np.random.normal(5.5, 1.2, 1)[0]
        happiness = max(2, min(8, happiness))
        
        if i % 10 == 0:  # 10% missing
            row['Happiness Score'] = np.random.choice(missing_patterns)
        elif i % 15 == 0:  # As string
            row['Happiness Score'] = f"{happiness:.2f} points"
        else:
            row['Happiness Score'] = round(happiness, 2)
        
        # GDP - different formats and missing values
        gdp = np.exp(np.random.normal(9, 1, 1)[0])
        gdp = max(500, min(120000, gdp))
        
        if i % 8 == 0:  # Missing
            row['GDP per capita'] = np.random.choice(missing_patterns)
        elif i % 12 == 0:  # String with thousands separator
            row['GDP per capita'] = f"${gdp:,.0f}"
        elif i % 20 == 0:  # Comma decimal
            row['GDP per capita'] = f"{gdp:.2f}".replace(".", ",")
        else:
            row['GDP per capita'] = round(gdp, 2)
        
        # Social support
        social = np.random.beta(2, 1, 1)[0] * 2
        if i % 7 == 0:
            row['Social support'] = np.random.choice(missing_patterns)
        else:
            row['Social support'] = round(social, 3)
        
        # Healthy life
        healthy = np.random.beta(2, 1, 1)[0] * 1.5
        if i % 9 == 0:
            row['Healthy life expectancy'] = np.random.choice(missing_patterns)
        else:
            row['Healthy life expectancy'] = round(healthy, 3)
        
        # Freedom
        freedom = np.random.beta(2, 1, 1)[0] * 0.8
        if i % 11 == 0:
            row['Freedom'] = np.random.choice(missing_patterns)
        else:
            row['Freedom'] = round(freedom, 3)
        
        # Generosity
        generosity = np.random.beta(1, 2, 1)[0] * 0.5
        if i % 13 == 0:
            row['Generosity'] = np.random.choice(missing_patterns)
        else:
            row['Generosity'] = round(generosity, 3)
        
        # Corruption perception
        corruption = np.random.beta(2, 2, 1)[0] * 0.6
        if i % 6 == 0:
            row['Corruption'] = np.random.choice(missing_patterns)
        else:
            row['Corruption'] = round(corruption, 3)
        
        messy_data.append(row)
    
    df = pd.DataFrame(messy_data)
    
    # Add extra columns to some rows (common in web scraping)
    if len(df) > 0:
        df['Extra Column'] = np.random.choice(['', None, 'extra'], len(df))
    
    print(f"Created messy data: {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_scraped_data(df_raw):
    """
    Cleans scraped messy data.
    This function demonstrates the step-by-step cleaning process.
    """
    print("\n" + "="*60)
    print("DATA CLEANING PROCESS")
    print("="*60)
    
    df = df_raw.copy()
    
    print(f"\n1. INITIAL STATE:")
    print(f"   - Number of rows: {len(df)}")
    print(f"   - Number of columns: {len(df.columns)}")
    print(f"   - Columns: {list(df.columns)}")
    
    # STEP 1: Remove unnecessary columns
    print(f"\n2. REMOVING UNNECESSARY COLUMNS:")
    cols_to_drop = [col for col in df.columns if 'Extra' in col or 'Unnamed' in str(col)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"   - Removed columns: {cols_to_drop}")
    
    # STEP 2: Clean country names (remove spaces)
    print(f"\n3. CLEANING COUNTRY NAMES:")
    if 'Country' in df.columns:
        df['Country'] = df['Country'].astype(str).str.strip()
        print(f"   - Spaces removed")
    
    # STEP 3: Clean numeric columns
    print(f"\n4. CLEANING NUMERIC COLUMNS:")
    
    numeric_columns = {
        'Happiness Score': ['Happiness Score', 'Score', 'Happiness'],
        'GDP per capita': ['GDP per capita', 'GDP', 'GDP_per_capita'],
        'Social support': ['Social support', 'Social Support'],
        'Healthy life expectancy': ['Healthy life expectancy', 'Life Expectancy'],
        'Freedom': ['Freedom', 'Freedom to make life choices'],
        'Generosity': ['Generosity'],
        'Corruption': ['Corruption', 'Perceptions of corruption']
    }
    
    for target_col, possible_names in numeric_columns.items():
        # Find column name
        col_name = None
        for name in possible_names:
            if name in df.columns:
                col_name = name
                break
        
        if col_name:
            print(f"\n   Cleaning {target_col}...")
            
            # Convert string to number
            def clean_numeric(value):
                if pd.isna(value) or value in [None, "N/A", "", "-", "n/a", "NULL", "null"]:
                    return np.nan
                
                # If string, clean it
                if isinstance(value, str):
                    # Remove characters like "points", "$", ","
                    value = re.sub(r'[points$,\s]', '', value)
                    # Convert comma to dot (decimal separator)
                    value = value.replace(',', '.')
                    # Extract only numbers
                    value = re.sub(r'[^\d.]', '', value)
                    
                    try:
                        return float(value)
                    except:
                        return np.nan
                
                return float(value)
            
            df[target_col] = df[col_name].apply(clean_numeric)
            
            # Remove old column (if different name)
            if col_name != target_col:
                df = df.drop(columns=[col_name])
    
    # STEP 4: Check and fill missing values
    print(f"\n5. CLEANING MISSING VALUES:")
    missing_before = df.isnull().sum().sum()
    print(f"   - Total missing values: {missing_before}")
    
    # Fill with mean for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Country':
            mean_val = df[col].mean()
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(mean_val)
            if missing_count > 0:
                print(f"   - {col}: {missing_count} missing values filled with mean ({mean_val:.2f})")
    
    # STEP 5: Clean outliers (too large/small)
    print(f"\n6. CLEANING OUTLIERS:")
    for col in numeric_cols:
        if col != 'Country':
            # Find outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 3*IQR wider threshold
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                # Clip outliers to thresholds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"   - {col}: {outliers} outliers corrected")
    
    # STEP 6: Remove duplicate rows
    print(f"\n7. REMOVING DUPLICATE ROWS:")
    duplicates = df.duplicated(subset='Country').sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset='Country', keep='first')
        print(f"   - {duplicates} duplicate rows removed")
    else:
        print(f"   - No duplicate rows")
    
    # STEP 7: Final check
    print(f"\n8. FINAL STATE:")
    print(f"   - Number of rows: {len(df)}")
    print(f"   - Number of columns: {len(df.columns)}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Cleaned columns: {list(df.columns)}")
    
    print("\n" + "="*60)
    print("DATA CLEANING COMPLETED")
    print("="*60 + "\n")
    
    return df


if __name__ == "__main__":
    # Test
    print("Testing web scraping...")
    raw_data = scrape_wikipedia_happiness_data()
    print(f"\nRaw data sample:")
    print(raw_data.head())
    print(f"\nRaw data info:")
    print(raw_data.info())
    
    cleaned_data = clean_scraped_data(raw_data)
    print(f"\nCleaned data sample:")
    print(cleaned_data.head())
    print(f"\nCleaned data statistics:")
    print(cleaned_data.describe())
