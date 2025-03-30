#!/usr/bin/env python3
"""
Test script to verify that the Python environment is set up correctly.
This checks all required dependencies and their versions.
"""

import sys
import importlib
import pkg_resources

def check_package(package_name):
    """Check if a package is installed and report its version."""
    module_name = package_name
    # Handle special cases where import name differs from package name
    if package_name == "scikit-learn":
        module_name = "sklearn"
    elif package_name == "dash_bootstrap_components":
        module_name = "dash_bootstrap_components"
    
    try:
        module = importlib.import_module(module_name)
        try:
            version = pkg_resources.get_distribution(package_name).version
            print(f"✅ {package_name} {version} - OK")
            return True
        except pkg_resources.DistributionNotFound:
            print(f"✅ {package_name} (version unknown) - OK")
            return True
    except ImportError:
        print(f"❌ {package_name} - NOT FOUND")
        return False

def main():
    """Main function to check all required packages."""
    print(f"Python version: {sys.version}")
    print("\nChecking required packages:")
    
    dependencies = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "yfinance",
        "statsmodels",
        "plotly",
        "dash",
        "dash_bootstrap_components",
        "jupyter",
        "nbformat"
    ]
    
    all_passed = True
    for dep in dependencies:
        if not check_package(dep):
            all_passed = False
    
    if all_passed:
        print("\nAll dependencies are installed correctly! 🎉")
        print("You can run the stock analysis tools using:")
        print("- Python script: python stock_analysis.py")
        print("- Web dashboard: python stock_dashboard.py")
        print("- Jupyter notebook: jupyter notebook Stock_predictor.ipynb")
    else:
        print("\n⚠️ Some dependencies are missing.")
        print("Please run: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 