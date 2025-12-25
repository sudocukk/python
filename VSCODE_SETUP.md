# VS Code Setup Guide

## 🚀 Quick Start

### Method 1: Using Batch File (Easiest)
1. Double-click the `open_in_vscode.bat` file
2. VS Code will automatically open and load the project

### Method 2: From VS Code Menu
1. Open VS Code
2. Click `File` → `Open Folder` menu (or `Ctrl+K Ctrl+O`)
3. Select this folder: `C:\Users\CASPER\Desktop\python`
4. Click the `Select Folder` button

### Method 3: From Command Line
1. Open Terminal or CMD
2. Run this command:
```bash
cd C:\Users\CASPER\Desktop\python
code .
```

### Method 4: Drag and Drop
1. Open `C:\Users\CASPER\Desktop\python` folder in Windows Explorer
2. Drag and drop the folder onto VS Code window

## 📦 Recommended VS Code Extensions

After opening the project, VS Code will show recommended extensions. I recommend installing:

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

## 🎯 First Steps

### 1. Select Python Interpreter
1. After VS Code opens, click the Python version in the bottom right corner
2. Or press `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Select a Python 3.8+ version

### 2. Install Dependencies
Open terminal in VS Code (`Ctrl+`` or `Terminal` → `New Terminal`) and run:

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook
1. Open `happiness_analysis.ipynb` file
2. Click the "Select Kernel" button in the top right
3. Select your Python interpreter
4. Use `Shift+Enter` to run cells

### 4. Run Web Application
In terminal:
```bash
python web_app.py
```
Then open `http://127.0.0.1:8050` in your browser

## 📁 Project Structure

```
python/
├── .vscode/                  # VS Code settings
│   ├── extensions.json       # Recommended extensions
│   └── settings.json         # Project settings
├── happiness_analysis.ipynb  # Main analysis notebook
├── data_scraper.py          # Web scraping module
├── anomaly_detection.py     # Anomaly detection
├── web_app.py               # Plotly Dash application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── open_in_vscode.bat       # Batch file to open VS Code
└── VSCODE_SETUP.md          # This file
```

## 🔧 VS Code Settings

The project includes these settings in `.vscode/settings.json`:
- Python auto import
- Format on save
- Jupyter kernel settings

## ❓ Troubleshooting

### Python Interpreter Not Found
1. Make sure Python is installed: `python --version`
2. In VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Select the correct Python path

### Jupyter Notebook Won't Open
1. Make sure Jupyter extension is installed
2. In terminal: `pip install jupyter`
3. Restart VS Code

### Dependencies Won't Install
1. If using virtual environment, activate it
2. Or: `python -m pip install -r requirements.txt`

### Web Application Won't Run
1. Port 8050 might be used by another application
2. Change the port number in `web_app.py` file
3. Or check with: `netstat -ano | findstr :8050`

## 📚 Extra Tips

- **Code Completion**: See code suggestions with `Ctrl+Space`
- **Command Palette**: Access all commands with `Ctrl+Shift+P`
- **Quick File Open**: Press `Ctrl+P` and type filename to open
- **Terminal**: Toggle terminal with `Ctrl+``
- **Split Editor**: Split editor with `Ctrl+\`

## 🎓 Learning Resources

- [VS Code Python Documentation](https://code.visualstudio.com/docs/languages/python)
- [Jupyter Notebook in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

Good luck! 🚀

