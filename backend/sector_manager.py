"""Dynamic JSE sector and ticker management pipeline module.

Replaces hardcoded lists and manual code-editing scripts with a dynamic data-driven pipeline.
Manages Top 40 constituents and 12-sector classifications loaded from JSON config or imported
from official constituent spreadsheets (e.g. Satrix constituent_details.xlsx).
"""
import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Default file locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_PATH = os.path.join(DATA_DIR, "jse_sectors.json")

# Standard 12 Industry Mapping Rules
INDUSTRY_MAPPING_RULES = {
    "Banks": "Banks",
    "Basic Resources": "Basic Resources",
    "Chemicals": "Chemicals",
    "Consumer Products and Services": "Consumer Products and Services",
    "Consumer Products & Services": "Consumer Products and Services",
    "Financial Services": "Financial Services",
    "Food Beverage and Tobacco": "Food Beverage and Tobacco",
    "Food, Beverage & Tobacco": "Food Beverage and Tobacco",
    "Industrial Goods & Sevices": "Industrial Goods & Sevices",
    "Industrial Goods & Services": "Industrial Goods & Sevices",
    "Insurance": "Insurance",
    "Personal Care Drug and Grocery Stores": "Personal Care Drug and Grocery Stores",
    "Personal Care, Drug & Grocery": "Personal Care Drug and Grocery Stores",
    "Real Estate": "Real Estate",
    "Retail": "Retail",
    "Technology": "Technology",
    "Telecommunications": "Telecommunications",
}

# In-memory runtime cache
_SECTOR_CACHE: Optional[Dict[str, Any]] = None


def _get_fallback_data() -> Dict[str, Any]:
    """Provides safe embedded fallback data if JSON file is missing."""
    return {
        "last_updated": "embedded_fallback",
        "top_40": [
            "ABG.JO", "AGL.JO", "ANG.JO", "ANH.JO", "APN.JO", "BHG.JO", "BID.JO", "BTI.JO", "BVT.JO", "CFR.JO",
            "CLS.JO", "CPI.JO", "DSY.JO", "EXX.JO", "FSR.JO", "GFI.JO", "GLN.JO", "GRT.JO", "HAR.JO", "IMP.JO",
            "INL.JO", "INP.JO", "MCG.JO", "MNP.JO", "MRP.JO", "MTN.JO", "NED.JO", "NPH.JO", "NPN.JO", "NRP.JO",
            "OMU.JO", "OUT.JO", "PAN.JO", "PPH.JO", "PRX.JO", "REM.JO", "RMI.JO", "RNI.JO", "SBK.JO", "SHP.JO",
            "SLM.JO", "SOL.JO", "SSW.JO", "VAL.JO", "VOD.JO", "WHL.JO"
        ],
        "sectors": {
            "ABG.JO": "Banks", "CPI.JO": "Banks", "FSR.JO": "Banks", "INL.JO": "Banks", "INP.JO": "Banks", "NED.JO": "Banks", "SBK.JO": "Banks",
            "AGL.JO": "Basic Resources", "ANG.JO": "Basic Resources", "BHG.JO": "Basic Resources", "EXX.JO": "Basic Resources",
            "GFI.JO": "Basic Resources", "GLN.JO": "Basic Resources", "HAR.JO": "Basic Resources", "IMP.JO": "Basic Resources",
            "NPH.JO": "Basic Resources", "PAN.JO": "Basic Resources", "SSW.JO": "Basic Resources", "VAL.JO": "Basic Resources",
            "SOL.JO": "Chemicals", "CFR.JO": "Consumer Products and Services", "REM.JO": "Financial Services", "RNI.JO": "Financial Services",
            "ANH.JO": "Food Beverage and Tobacco", "BTI.JO": "Food Beverage and Tobacco",
            "BVT.JO": "Industrial Goods & Sevices", "MNP.JO": "Industrial Goods & Sevices",
            "DSY.JO": "Insurance", "OMU.JO": "Insurance", "OUT.JO": "Insurance", "RMI.JO": "Insurance", "SLM.JO": "Insurance",
            "BID.JO": "Personal Care Drug and Grocery Stores", "CLS.JO": "Personal Care Drug and Grocery Stores", "SHP.JO": "Personal Care Drug and Grocery Stores",
            "GRT.JO": "Real Estate", "NRP.JO": "Real Estate",
            "APN.JO": "Retail", "MRP.JO": "Retail", "PPH.JO": "Retail", "WHL.JO": "Retail",
            "MCG.JO": "Technology", "NPN.JO": "Technology", "PRX.JO": "Technology",
            "MTN.JO": "Telecommunications", "VOD.JO": "Telecommunications"
        }
    }


def load_sector_data(force_reload: bool = False) -> Dict[str, Any]:
    """Loads sector and ticker data from backend JSON store into runtime memory cache."""
    global _SECTOR_CACHE
    if _SECTOR_CACHE is not None and not force_reload:
        return _SECTOR_CACHE

    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "top_40" in data and "sectors" in data:
                    _SECTOR_CACHE = data
                    logger.info(f"Loaded sector metadata from {JSON_PATH}")
                    return _SECTOR_CACHE
        except Exception as e:
            logger.error(f"Failed to parse sector JSON at {JSON_PATH}: {e}")

    logger.warning("Using fallback sector data")
    _SECTOR_CACHE = _get_fallback_data()
    return _SECTOR_CACHE


def save_sector_data(data: Dict[str, Any]) -> bool:
    """Saves sector metadata back to JSON store and updates cache."""
    global _SECTOR_CACHE
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        _SECTOR_CACHE = data
        logger.info(f"Successfully saved updated sector data to {JSON_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to save sector data to {JSON_PATH}: {e}")
        return False


def get_jse_top_40() -> List[str]:
    """Returns the active list of JSE Top 40 tickers."""
    data = load_sector_data()
    return data.get("top_40", []).copy()


def get_jse_sectors() -> Dict[str, str]:
    """Returns the dict of ticker -> sector mappings."""
    data = load_sector_data()
    return data.get("sectors", {}).copy()


def get_sector_for_ticker(ticker: str, fallback_yfinance: bool = True) -> str:
    """Gets the sector for a ticker. Fallbacks to yfinance info if enabled."""
    sectors = get_jse_sectors()
    if ticker in sectors:
        return sectors[ticker]

    if fallback_yfinance:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info
            if info and info.get('sector'):
                return info['sector']
        except Exception as e:
            logger.debug(f"YFinance sector lookup failed for {ticker}: {e}")

    return "Other"


def import_from_excel(excel_path: str) -> bool:
    """
    Parses a Satrix/JSE constituent Excel spreadsheet (e.g. constituent_details.xlsx)
    and dynamically updates sector mappings & Top 40 tickers without editing python source files.
    """
    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found at: {excel_path}")
        return False

    try:
        import pandas as pd
        # Try reading sheet with header auto-detection
        df = pd.read_excel(excel_path, skiprows=2, names=['Code', 'Company', 'Industry', 'Shares', 'DivYield', 'MarketCap', 'Price'])
        
        # Clean up header row if included
        if 'Industry' in df.columns:
            df = df[df['Industry'] != 'INDUSTRY'].dropna(subset=['Code', 'Industry'])

        if df.empty:
            logger.error("Parsed DataFrame from Excel is empty")
            return False

        tickers = []
        sectors = {}
        from datetime import datetime

        for _, row in df.iterrows():
            code = str(row['Code']).strip()
            if not code or code == 'nan':
                continue
            ticker = f"{code}.JO" if not code.endswith(".JO") else code
            raw_industry = str(row['Industry']).strip()
            
            # Map raw industry to normalized 12 industries
            sector = INDUSTRY_MAPPING_RULES.get(raw_industry, raw_industry)
            
            tickers.append(ticker)
            sectors[ticker] = sector

        tickers = sorted(list(set(tickers)))

        new_data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": f"Imported from {os.path.basename(excel_path)}",
            "top_40": tickers,
            "sectors": sectors
        }

        success = save_sector_data(new_data)
        if success:
            logger.info(f"Imported {len(tickers)} tickers and {len(set(sectors.values()))} sectors from {excel_path}")
        return success
    except Exception as e:
        logger.error(f"Error importing from Excel {excel_path}: {e}", exc_info=True)
        return False


def sync_sector_data(auto_search_excel: bool = True) -> Dict[str, Any]:
    """
    Triggers dynamic synchronization.
    If an updated Excel spreadsheet is present in standard locations, parses and syncs it.
    """
    search_paths = [
        os.path.join(os.path.expanduser("~"), "Downloads", "constituent_details.xlsx"),
        os.path.join(DATA_DIR, "constituent_details.xlsx"),
        os.path.join(BASE_DIR, "constituent_details.xlsx"),
    ]

    synced_from = None
    if auto_search_excel:
        for path in search_paths:
            if os.path.exists(path):
                if import_from_excel(path):
                    synced_from = path
                    break

    data = load_sector_data(force_reload=True)
    return {
        "status": "success",
        "synced_from_excel": synced_from,
        "total_tickers": len(data.get("top_40", [])),
        "total_sectors": len(set(data.get("sectors", {}).values())),
        "last_updated": data.get("last_updated"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSE Sector & Ticker Pipeline Manager")
    parser.add_argument("--import-excel", type=str, help="Path to constituent_details.xlsx to import")
    parser.add_argument("--sync", action="store_true", help="Sync sector data from available sources")
    parser.add_argument("--show", action="store_true", help="Display current sector summary")

    args = parser.parse_args()

    if args.import_excel:
        res = import_from_excel(args.import_excel)
        print(f"Import result: {'SUCCESS' if res else 'FAILED'}")
    elif args.sync:
        res = sync_sector_data()
        print(f"Sync result: {res}")
    elif args.show:
        d = load_sector_data()
        print(f"Top 40 count: {len(d['top_40'])}")
        print(f"Sectors mapping count: {len(d['sectors'])}")
        print(f"Unique sectors: {sorted(list(set(d['sectors'].values())))}")
    else:
        parser.print_help()
