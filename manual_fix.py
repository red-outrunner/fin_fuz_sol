#!/usr/bin/env python3
"""Manually fix screener.py with exact 12 Satrix industries"""

# Read the file
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'r') as f:
    lines = f.readlines()

# Find and replace JSE_TOP_40
new_top40 = '''# === JSE Top 40 Tickers (from Satrix constituent_details.xlsx) ===
JSE_TOP_40 = [
    "ABG.JO",  # Absa Group Ltd - Banks
    "AGL.JO",  # Anglo American - Basic Resources
    "ANG.JO",  # AngloGold Ashanti - Basic Resources
    "ANH.JO",  # Anheuser-Busch InBev - Food Beverage and Tobacco
    "APN.JO",  # Aspen Pharmacare - Retail
    "BHG.JO",  # BHP Group Limited - Basic Resources
    "BID.JO",  # Bid Corp Ltd - Personal Care Drug and Grocery Stores
    "BTI.JO",  # British American Tobacco - Food Beverage and Tobacco
    "BVT.JO",  # Bidvest Group Limited - Industrial Goods & Sevices
    "CFR.JO",  # Compagnie Financiere - Consumer Products and Services
    "CLS.JO",  # Clicks Group - Personal Care Drug and Grocery Stores
    "CPI.JO",  # Capitec - Banks
    "DSY.JO",  # Discovery - Insurance
    "EXX.JO",  # Exxaro Resources - Basic Resources
    "FSR.JO",  # Firstrand - Banks
    "GFI.JO",  # Gold Fields Ltd - Basic Resources
    "GLN.JO",  # Glencore Plc - Basic Resources
    "GRT.JO",  # Growthpoint - Real Estate
    "HAR.JO",  # Harmony Gold Mining - Basic Resources
    "IMP.JO",  # Implats - Basic Resources
    "INL.JO",  # Investec Ltd - Banks
    "INP.JO",  # Investec Plc - Banks
    "MCG.JO",  # MultiChoice Group - Technology
    "MNP.JO",  # Murray & Roberts - Industrial Goods & Sevices
    "MRP.JO",  # Mr Price Group Ltd - Retail
    "MTN.JO",  # MTN Group - Telecommunications
    "NED.JO",  # Nedbank - Banks
    "NPH.JO",  # Northam Platinum - Basic Resources
    "NPN.JO",  # Naspers - Technology
    "NRP.JO",  # NEPI Rockcastle Plc - Real Estate
    "OMU.JO",  # Old Mutual Ltd - Insurance
    "OUT.JO",  # Outsurance - Insurance
    "PAN.JO",  # Pan African Resources - Basic Resources
    "PPH.JO",  # Pepkor Holdings Ltd - Retail
    "PRX.JO",  # Prosus Nv - Technology
    "REM.JO",  # Remgro - Financial Services
    "RMI.JO",  # Rand Merchant Investment - Insurance
    "RNI.JO",  # Reinet Investments Sca - Financial Services
    "SBK.JO",  # Standard Bank - Banks
    "SHP.JO",  # Shoprite - Personal Care Drug and Grocery Stores
    "SLM.JO",  # Sanlam - Insurance
    "SOL.JO",  # Sasol - Chemicals
    "SSW.JO",  # Sibanye Stillwater Ltd - Basic Resources
    "VAL.JO",  # Valterra Platinum Ltd - Basic Resources
    "VOD.JO",  # Vodacom Group Limited - Telecommunications
    "WHL.JO",  # Woolworths Holdings Ltd - Retail
]
'''

# Find JSE_TOP_40 start and JSE_SECTORS end
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# === JSE Top 40' in line or 'JSE_TOP_40 = [' in line:
        start_idx = i if start_idx is None else start_idx
    if start_idx is not None and line.strip() == ']' and i > start_idx:
        # Found end of JSE_TOP_40, now find JSE_SECTORS
        for j in range(i, min(i+400, len(lines))):
            if 'JSE_SECTORS = {' in lines[j]:
                # Find closing brace
                for k in range(j, min(j+200, len(lines))):
                    if lines[k].strip() == '}':
                        end_idx = k + 1
                        break
                break
        break

if start_idx and end_idx:
    # Replace the section
    new_lines = lines[:start_idx] + [new_top40 + '\n\n'] + lines[end_idx:]
    
    with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Updated JSE_TOP_40 and JSE_SECTORS with 12 Satrix industries")
else:
    print(f"❌ Could not find markers. start={start_idx}, end={end_idx}")
