import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    """
    Read portfolio data from CSV or Excel file selected by user
    Returns a pandas DataFrame
    """
    # Create root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Portfolio File",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
            ("All files", "*.*")
        ]
    )
    
    # Destroy the root window
    root.destroy()
    
    if not file_path:
        raise FileNotFoundError("No file selected")
    
    # Read file based on extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")
    
    # Validate required columns
    required_columns = ['Asset Name', 'Type', 'Value (R)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Expected: {required_columns}")
    
    return df

def calculate_allocation(df):
    """
    Group assets by type and calculate total per category
    Calculate portfolio total and % allocation per category
    """
    # Group by type and sum values
    allocation = df.groupby('Type')['Value (R)'].sum().reset_index()
    
    # Calculate total portfolio value
    total_value = allocation['Value (R)'].sum()
    
    # Calculate percentage allocation
    allocation['Percentage'] = (allocation['Value (R)'] / total_value * 100).round(2)
    
    return allocation, total_value

def calculate_drift(allocation, targets={ 'Income': 40, 'Equity': 40, 'Alternative': 20 }):
    """
    Compare with target allocation and compute drift
    """
    # Merge with target allocations
    target_df = pd.DataFrame(list(targets.items()), columns=['Type', 'Target'])
    drift_df = pd.merge(allocation, target_df, on='Type', how='right')
    drift_df['Drift'] = drift_df['Percentage'] - drift_df['Target']
    drift_df['Drift'] = drift_df['Drift'].round(2)
    
    return drift_df

def suggest_rebalance(drift_df, new_capital, total_value):
    """
    Recommend allocation of new capital to minimize drift
    """
    # Calculate the ideal allocation amounts
    targets = dict(zip(drift_df['Type'], drift_df['Target']))
    ideal_allocations = {k: (v/100) * (total_value + new_capital) for k, v in targets.items()}
    
    # Calculate current allocations
    current_allocations = dict(zip(drift_df['Type'], drift_df['Value (R)']))
    
    # Calculate how much to add to each category
    additional_allocations = {}
    for category in targets.keys():
        additional_allocations[category] = max(0, ideal_allocations[category] - current_allocations.get(category, 0))
    
    # Normalize to ensure the sum equals new_capital
    total_suggested = sum(additional_allocations.values())
    if total_suggested > 0:
        adjustment_factor = new_capital / total_suggested
        for category in additional_allocations:
            additional_allocations[category] *= adjustment_factor
    
    # Round to nearest whole number
    for category in additional_allocations:
        additional_allocations[category] = round(additional_allocations[category])
    
    return additional_allocations

def print_summary(df, allocation, drift_df, new_capital, rebalance_suggestion, total_value):
    """
    Print a summary of the portfolio analysis
    """
    print("="*60)
    print("PORTFOLIO ALLOCATION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal Portfolio Value: R{total_value:,.2f}")
    print(f"New Capital to Allocate: R{new_capital:,.2f}")
    print(f"Total Value After Allocation: R{total_value + new_capital:,.2f}")
    
    print("\nCurrent vs Target Allocation:")
    print("-" * 50)
    for _, row in drift_df.iterrows():
        status = "Overweight" if row['Drift'] > 0 else "Underweight" if row['Drift'] < 0 else "On Target"
        print(f"{row['Type']:>12}: {row['Percentage']:>6.2f}% (Target: {row['Target']:>6.2f}%) [{status}]")
    
    print("\nSuggested Capital Allocation:")
    print("-" * 50)
    for category, amount in rebalance_suggestion.items():
        print(f"{category:>12}: R{amount:>8,}")
    
    print("\nDetailed Asset Holdings:")
    print("-" * 50)
    for asset_type in df['Type'].unique():
        print(f"\n{asset_type}:")
        assets = df[df['Type'] == asset_type]
        for _, asset in assets.iterrows():
            print(f"  - {asset['Asset Name']}: R{asset['Value (R)']:,.2f}")

def plot_allocation(allocation, drift_df, rebalance_suggestion, total_value, new_capital):
    """
    Create pie charts showing before and after rebalancing
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Before rebalancing
    ax1.pie(allocation['Value (R)'], labels=allocation['Type'], autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Current Allocation (R{total_value:,.0f})')
    
    # After rebalancing
    # Calculate new values after adding suggested capital
    new_values = []
    types = []
    for _, row in drift_df.iterrows():
        types.append(row['Type'])
        current_value = row['Value (R)']
        additional = rebalance_suggestion.get(row['Type'], 0)
        new_values.append(current_value + additional)
    
    ax2.pie(new_values, labels=types, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Allocation After Adding R{new_capital:,.0f}')
    
    plt.tight_layout()
    plt.show()

def get_custom_targets():
    """
    Allow user to input custom allocation targets
    """
    print("Enter custom allocation targets (must sum to 100%):")
    income = float(input("Income target (%): "))
    equity = float(input("Equity target (%): "))
    alternative = float(input("Alternative target (%): "))
    
    if abs(income + equity + alternative - 100) > 0.01:
        print("Warning: Targets do not sum to 100%. Using default 40/40/20 allocation.")
        return {'Income': 40, 'Equity': 40, 'Alternative': 20}
    
    return {'Income': income, 'Equity': equity, 'Alternative': alternative}

def main():
    try:
        # Read data
        df = read_data()
        
        # Calculate allocation
        allocation, total_value = calculate_allocation(df)
        
        # Get targets (custom or default)
        use_custom = input("Use custom allocation targets? (y/n): ").lower().startswith('y')
        if use_custom:
            targets = get_custom_targets()
        else:
            targets = {'Income': 40, 'Equity': 40, 'Alternative': 20}
        
        # Calculate drift
        drift_df = calculate_drift(allocation, targets)
        
        # Get new capital amount
        new_capital = float(input("Enter new capital to allocate (R): "))
        
        # Suggest rebalance
        rebalance_suggestion = suggest_rebalance(drift_df, new_capital, total_value)
        
        # Print summary
        print_summary(df, allocation, drift_df, new_capital, rebalance_suggestion, total_value)
        
        # Plot allocation
        plot = input("\nShow pie charts? (y/n): ").lower().startswith('y')
        if plot:
            plot_allocation(allocation, drift_df, rebalance_suggestion, total_value, new_capital)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
