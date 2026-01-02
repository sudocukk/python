# Global Happiness and Economic Prosperity Analysis 🌍

This project examines the relationship between countries' happiness scores and economic and social indicators using the World Happiness Report dataset.

## 📋 Project Contents

### Basic Features
- **Data Processing**: Data cleaning, merging, and grouping with Pandas
- **Visualization**: Various graphs with Matplotlib and Seaborn
- **Statistical Analysis**: Correlation analysis and hypothesis tests

### Advanced Features (Extra Points)
- ✅ **Web Scraping**: Scraping messy data from the web (`data_scraper.py`)
- ✅ **Anomaly Detection**: IQR, Z-Score, and Isolation Forest methods (`anomaly_detection.py`)
- ✅ **Web-Based Visualization**: Interactive web application with Plotly Dash (`web_app.py`)
- ✅ **Advanced Statistical Reporting**: Quartiles, box plots, skewness, kurtosis analyses

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Jupyter Notebook Analysis

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open `happiness_analysis.ipynb` and run the cells sequentially.

### 2. Web-Based Visualization Application

Start the interactive web application:

```bash
python web_app.py
```

Open `http://127.0.0.1:8050` in your browser.

## 🔍 Dataset and Web Scraping

This project uses **web scraping** to fetch data. The `data_scraper.py` module:

1. Fetches data from Wikipedia or other web sources
2. Generates **messy data** (missing values, different formats, special characters)
3. Cleans the data step by step

### Data Cleaning Process (Step by Step)

#### STEP 1: Removing Unnecessary Columns
- Unnecessary columns like "Extra Column", "Unnamed" are detected and removed
- Extra columns commonly seen in web scraping are cleaned

#### STEP 2: Cleaning Country Names
- Extra spaces are removed using `strip()`
- Example: `"  Finland  "` → `"Finland"`

#### STEP 3: Cleaning Numeric Columns
- String values are converted to numeric values
- Special characters are cleaned: `"7.5 points"` → `7.5`
- Currency symbols are removed: `"$50,000"` → `50000`
- Decimal separators are normalized: `"7,5"` → `7.5` (comma converted to dot)

#### STEP 4: Cleaning Missing Values
- Missing value patterns are detected: `None`, `"N/A"`, `""`, `"-"`, `"NULL"`, `"n/a"`
- Missing values in numeric columns are filled with the column mean
- Missing value count is reported

#### STEP 5: Cleaning Outliers
- Outliers are detected using IQR (Interquartile Range) method
- Lower threshold: `Q1 - 3*IQR`
- Upper threshold: `Q3 + 3*IQR`
- Outliers are clipped to thresholds

#### STEP 6: Removing Duplicate Rows
- Duplicate rows are detected by country name
- First record is kept, others are removed

#### STEP 7: Final Check
- Cleaned dataset is checked
- Final row/column counts are reported
- Missing value count is verified

### Example Messy Data Problems and Solutions

| Problem | Example | Solution |
|---------|---------|----------|
| Extra spaces | `"  Finland  "` | `.strip()` |
| Missing values | `"N/A"`, `"-"`, `""` | `fillna(mean)` |
| String format | `"7.5 points"` | Extract numbers with regex |
| Currency | `"$50,000"` | Remove special characters |
| Comma decimal | `"7,5"` | Convert comma to dot |
| Outlier | `GDP: 2000000` | Detect and clip with IQR |
| Duplicate | Two "Finland" entries | `drop_duplicates()` |

## 📊 Analysis Scope

### Basic Analyses
1. Correlation between economy and happiness (Scatter Plot)
2. Comparison of happiest and unhappiest countries (Bar Chart)
3. Impact of factors on happiness (Heatmap)
4. Regional comparisons (Analysis by continents)
5. Changes over time

### Advanced Analyses
6. **Anomaly Detection**:
   - Univariate anomaly detection with IQR method
   - Statistical anomaly detection with Z-Score method
   - Multivariate anomaly detection with Isolation Forest

7. **Advanced Statistical Reporting**:
   - Quartiles (Q1, Q2, Q3) and IQR calculations
   - Box and Whisker Plots
   - Skewness analysis
   - Kurtosis analysis
   - Normal distribution comparisons

## 📁 File Structure

```
python/
├── happiness_analysis.ipynb    # Main analysis notebook
├── data_scraper.py              # Web scraping and data cleaning
├── anomaly_detection.py         # Anomaly detection module
├── web_app.py                   # Plotly Dash web application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Git ignore file
```

## 🔧 Requirements

- Python 3.8+
- All dependencies are listed in `requirements.txt` file

## 📝 Notes

- If web scraping fails, sample data will be automatically generated
- Data cleaning process automatically reports all steps
- Anomaly detection results are presented with detailed reports


### Required Extensions:
1. **Python** (ms-python.python)
   - Required for Python support
   
2. **Jupyter** (ms-toolsai.jupyter)
   - To run `.ipynb` files

3. **Pylance** (ms-python.vscode-pylance)
   - Advanced IntelliSense for Python

### Optional (But Recommended):
4. **Black Formatter** (ms-python.black-formatter)
   - For code formatting

5. **GitLens** (eamodio.gitlens)
   - For Git version control