#!/usr/bin/env python3
"""Direct update of screener.py with 12 Satrix industries"""

content = """# === JSE Top 40 Tickers (from Satrix constituent_details.xlsx) ===
JSE_TOP_40 = [
    "ABG.JO",  # Absa Group
    "AGL.JO",  # Anglo American
    "ANG.JO",  # AngloGold Ashanti
    "ANH.JO",  # Anheuser-Busch InBev
    "APN.JO",  # Aspen Pharmacare
    "BHG.JO",  # BHP Group
    "BID.JO",  # Bid Corporation
    "BTI.JO",  # British American Tobacco
    "BVT.JO",  # Bidvest
    "CFR.JO",  # Compagnie Financiere
    "CLS.JO",  # Clicks Group
    "CPI.JO",  # Capitec
    "DSY.JO",  # Discovery
    "EXX.JO",  # Exxaro Resources
    "FSR.JO",  # FirstRand
    "GFI.JO",  # Gold Fields
    "GLN.JO",  # Glencore
    "GRT.JO",  # Growthpoint Properties
    "HAR.JO",  # Harmony Gold
    "IMP.JO",  # Implats Platinum
    "INL.JO",  # Investec
    "INP.JO",  # Investec Plc
    "MCG.JO",  # MultiChoice
    "MNP.JO",  # Murray & Roberts
    "MRP.JO",  # Mr Price
    "MTN.JO",  # MTN Group
    "NED.JO",  # Nedbank
    "NPH.JO",  # Northam Platinum
    "NPN.JO",  # Naspers
    "NRP.JO",  # NEPI Rockcastle
    "OMU.JO",  # Old Mutual
    "OUT.JO",  # Outsurance
    "PAN.JO",  # Pan African Resources
    "PPH.JO",  # Pepkor Holdings
    "PRX.JO",  # Prosus
    "REM.JO",  # Remgro
    "RMI.JO",  # Rand Merchant Investment
    "RNI.JO",  # Reinet Investments
    "SBK.JO",  # Standard Bank
    "SHP.JO",  # Shoprite
    "SLM.JO",  # Sanlam
    "SOL.JO",  # Sasol
    "SSW.JO",  # Sibanye Stillwater
    "VAL.JO",  # Valterra Platinum
    "VOD.JO",  # Vodacom
    "WHL.JO",  # Woolworths
]

# Sector mappings from Satrix constituent_details.xlsx - 12 Industries
JSE_SECTORS = {
    # Banks (4)
    "ABG.JO": "Banks",
    "CPI.JO": "Banks",
    "FSR.JO": "Banks",
    "INL.JO": "Banks",
    "INP.JO": "Banks",
    "NED.JO": "Banks",
    "SBK.JO": "Banks",
    
    # Basic Resources (11)
    "AGL.JO": "Basic Resources",
    "ANG.JO": "Basic Resources",
    "BHG.JO": "Basic Resources",
    "EXX.JO": "Basic Resources",
    "GFI.JO": "Basic Resources",
    "GLN.JO": "Basic Resources",
    "HAR.JO": "Basic Resources",
    "IMP.JO": "Basic Resources",
    "NPH.JO": "Basic Resources",
    "SSW.JO": "Basic Resources",
    "VAL.JO": "Basic Resources",
    
    # Chemicals (1)
    "SOL.JO": "Chemicals",
    
    # Consumer Products & Services (1)
    "CFR.JO": "Consumer Products & Services",
    
    # Financial Services (2)
    "REM.JO": "Financial Services",
    "RNI.JO": "Financial Services",
    
    # Food, Beverage & Tobacco (2)
    "ANH.JO": "Food, Beverage & Tobacco",
    "BTI.JO": "Food, Beverage & Tobacco",
    
    # Industrial Goods & Services (1)
    "BVT.JO": "Industrial Goods & Services",
    
    # Insurance (4)
    "DSY.JO": "Insurance",
    "OMU.JO": "Insurance",
    "OUT.JO": "Insurance",
    "SLM.JO": "Insurance",
    
    # Personal Care, Drug & Grocery (3)
    "BID.JO": "Personal Care, Drug & Grocery",
    "CLS.JO": "Personal Care, Drug & Grocery",
    "SHP.JO": "Personal Care, Drug & Grocery",
    
    # Real Estate (2)
    "GRT.JO": "Real Estate",
    "NRP.JO": "Real Estate",
    
    # Retail (3)
    "APN.JO": "Retail",
    "MRP.JO": "Retail",
    "PPH.JO": "Retail",
    "WHL.JO": "Retail",
    
    # Technology (2)
    "MCG.JO": "Technology",
    "NPN.JO": "Technology",
    "PRX.JO": "Technology",
    
    # Telecommunications (2)
    "MTN.JO": "Telecommunications",
    "VOD.JO": "Telecommunications",
    
    # Other (2) - MNP, RMI
    "MNP.JO": "Industrial Goods & Services",
    "RMI.JO": "Insurance",
}
"""

# Read current screener.py
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'r') as f:
    content_old = f.read()

# Find and replace the JSE_TOP_40 and JSE_SECTORS sections
import re

# Replace JSE_TOP_40
content_new = re.sub(
    r'# === JSE Top 40.*?\]',
    content.split(']')[0] + ']',
    content_old,
    flags=re.DOTALL
)

# Replace JSE_SECTORS
content_new = re.sub(
    r'# Sector mappings.*?^}',
    content.split('# ===')[0].strip() + '\n' + content.split('# Sector mappings')[1].split('\n\n')[0],
    content_new,
    flags=re.DOTALL | re.MULTILINE
)

# Write back
with open('/home/red/projects/active/fin_fuz_sol/backend/screener.py', 'w') as f:
    f.write(content_new)

print("✅ Updated with 12 Satrix industries")
