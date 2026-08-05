"""Market data access layer: yfinance fetching, disk caching, news/article
scraping, and the SSRF guard. Pure data-retrieval — no analytics/quant math."""
import os
import sys
import re
import time
import socket
import hashlib
import ipaddress
import logging
import urllib.parse
import concurrent.futures

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# core_math lives in the repo root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_math
from serialization import clean_data

logger = logging.getLogger(__name__)

# Parquet disk cache (portable + no arbitrary-code-execution risk of pickle).
CACHE_DIR = os.getenv("CACHE_DIR", "cache_v3")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h default
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(ticker, start_date, end_date):
    """Generates a unique cache filename based on request parameters."""
    raw = f"{ticker}_{start_date}_{end_date}"
    hashed = hashlib.md5(raw.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.parquet")


def download_data(ticker: str, start_date: str, end_date: str):
    """Downloads data from yfinance with a parquet disk cache (TTL: CACHE_TTL_SECONDS)."""
    cache_path = _get_cache_path(ticker, start_date, end_date)

    # Check cache (fresh within TTL).
    if os.path.exists(cache_path):
        try:
            if time.time() - os.path.getmtime(cache_path) < CACHE_TTL_SECONDS:
                return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

    try:
        # Use Ticker object to avoid shared state/caching issues with yf.download in threads
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)

        if data is None or data.empty:
            return None

        # Ensure index is datetime
        data.index = pd.to_datetime(data.index)

        # Save to cache
        try:
            data.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

        return data
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None


def fetch_multiple_tickers(tickers: list, start_date: str, end_date: str):
    """Fetches data for multiple tickers in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(download_data, ticker, start_date, end_date): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data is not None:
                    results[ticker] = data
            except Exception as e:
                logger.error(f"Parallel fetch error for {ticker}: {e}")
    return results


def normalize_dividend_yield(raw):
    """
    yfinance reports dividendYield inconsistently across versions/tickers: sometimes a
    fraction (0.034 = 3.4%) and sometimes a percentage (3.4 = 3.4%). Normalize to a
    fraction. Values > 1 are assumed to be percentages (a >100% yield is implausible).
    Returns None for missing/invalid input, 0.0 for a genuine zero.
    """
    if raw is None:
        return None
    try:
        y = float(raw)
    except (TypeError, ValueError):
        return None
    if y < 0:
        return None
    if y > 1:
        y = y / 100.0
    return y


def get_dividend_yield(ticker: str) -> float:
    """Fetches a normalized annual dividend yield (fraction) for a ticker, or 0.0."""
    try:
        t = yf.Ticker(ticker)
        y = normalize_dividend_yield(t.info.get('dividendYield'))
        return y if y else 0.0
    except Exception as e:
        logger.warning(f"Could not fetch dividend yield for {ticker}: {e}")
        return 0.0


# Largest-shareholder data for major JSE stocks, compiled from company annual reports /
# investor-relations disclosures. yfinance only exposes US SEC 13F filings, which for
# JSE tickers surface tiny foreign funds (e.g. a US EM index fund at ~1%) rather than
# the real largest holders — so we use this curated table for .JO names instead.
# Percentages are approximate and dated; refresh from the latest annual reports.
# (The PIC — Public Investment Corporation — is genuinely the largest holder of most
# JSE blue chips as SA's state pension-fund manager.)
JSE_MAJOR_SHAREHOLDERS = {
    # === JSE / Satrix Top 40 largest shareholders ===
    # Sourced from company annual reports / investor-relations / regulatory filings
    # (2024-2025); percentages are approximate and dated — refresh periodically. The
    # PIC (Public Investment Corporation, SA's ~R2.7tn state pension-fund manager) is
    # genuinely the largest holder of many JSE blue chips. percent=None means the name
    # is known but a clean single % wasn't available.

    # Banks & financials
    "SBK.JO": {"name": "ICBC (Ind. & Comm. Bank of China)",   "percent": 0.196, "as_of": "2024"},
    "FSR.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.16,  "as_of": "2024"},
    "NED.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.147, "as_of": "2025"},
    "ABG.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.14,  "as_of": "2025"},
    "CPI.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.16,  "as_of": "2025"},
    "INL.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.17,  "as_of": "2025"},
    "INP.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.14,  "as_of": "2025"},
    # Insurers
    "SLM.JO": {"name": "Ubuntu-Botho Investments",           "percent": 0.14,  "as_of": "2025"},
    "OMU.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.19,  "as_of": "2025"},
    "DSY.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.13,  "as_of": "2025"},
    # Tech / media / telco
    "NPN.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.19,  "as_of": "2025"},
    "PRX.JO": {"name": "Naspers",                             "percent": 0.57,  "as_of": "2024"},
    "MTN.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2025"},
    "VOD.JO": {"name": "Vodafone Group",                      "percent": 0.651, "as_of": "2024"},
    "MCG.JO": {"name": "Canal+ (Vivendi)",                    "percent": 0.94,  "as_of": "2025"},
    # Resources
    "SOL.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.171, "as_of": "2024"},
    "ANG.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.15,  "as_of": "2025"},
    "IMP.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2025"},
    "GFI.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2024"},
    "VAL.JO": {"name": "Anglo American",                      "percent": 0.199, "as_of": "2025"},
    "EXX.JO": {"name": "Eyesizwe RF (B-BBEE)",                "percent": 0.308, "as_of": "2025"},
    "NPH.JO": {"name": "Public Investment Corporation (PIC)", "percent": None,  "as_of": "2025"},
    "AGL.JO": {"name": "BlackRock",                           "percent": 0.084, "as_of": "2024"},
    "BHG.JO": {"name": "BlackRock",                           "percent": 0.07,  "as_of": "2025"},
    "GLN.JO": {"name": "Ivan Glasenberg (former CEO)",        "percent": None,  "as_of": "2025"},
    "MNP.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.10,  "as_of": "2025"},
    # Consumer / retail / industrial
    "SHP.JO": {"name": "Public Investment Corporation (PIC)", "percent": None,  "as_of": "2025"},
    "WHL.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2025"},
    "MRP.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2025"},
    "CLS.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2025"},
    "APN.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.20,  "as_of": "2025"},
    "BVT.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.22,  "as_of": "2025"},
    "BID.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.165, "as_of": "2025"},
    "GRT.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.21,  "as_of": "2025"},
    "ANH.JO": {"name": "Stichting AB InBev (founding families)", "percent": 0.335, "as_of": "2025"},
    "BTI.JO": {"name": "Capital Group (largest institution)", "percent": None,  "as_of": "2025"},
    # Rupert-family investment holdings
    "REM.JO": {"name": "Rupert family (Remgro control structure)", "percent": None, "as_of": "2025"},
    "RNI.JO": {"name": "Anton Rupert Trust (Rupert family)",  "percent": 0.249, "as_of": "2025"},
    # Remaining Top 40 (NRP - NEPI Rockcastle, RMH - RMB Holdings) intentionally return
    # N/A: both are widely held with no single dominant shareholder reliably sourced.
}


def get_biggest_shareholder(ticker: str, t):
    """Returns the largest shareholder, or None when we lack trustworthy data (so we
    never display a false name).

    - JSE (.JO): use the curated table (yfinance's US-13F data is wrong for the JSE).
    - US-listed (no exchange suffix): yfinance institutional_holders IS real 13F data.
    - Any other non-US ticker: return None ('not available') rather than 13F junk.
    """
    tk = (ticker or "").upper().strip()

    curated = JSE_MAJOR_SHAREHOLDERS.get(tk)
    if curated:
        return {"name": curated["name"], "percent": curated.get("percent"),
                "as_of": curated.get("as_of"), "source": "Company reports"}

    # institutional_holders is US SEC 13F data — only reliable for US tickers.
    if "." not in tk and not tk.startswith("^"):
        try:
            ih = t.institutional_holders
            if ih is not None and not ih.empty and "Holder" in ih.columns:
                top = ih.iloc[0]
                pct = top.get("pctHeld")
                return {
                    "name": str(top["Holder"]),
                    "percent": float(pct) if pd.notna(pct) else None,
                    "as_of": None,
                    "source": "SEC 13F",
                }
        except Exception as e:
            logger.warning(f"institutional_holders unavailable for {tk}: {e}")

    return None


def get_company_profile(ticker: str):
    """
    Fetches company profile: major shareholder and sentiment.
    Returns None if data unavailable (e.g. indices).
    """
    try:
        t = yf.Ticker(ticker)

        # 1. Biggest Shareholder — curated JSE data + real US 13F; None when unreliable.
        biggest_holder = get_biggest_shareholder(ticker, t)

        # 2. Sentiment / Recommendation
        sentiment = None
        try:
            info = t.info
            rec_key = info.get('recommendationKey') # 'buy', 'hold', 'sell', 'strong_buy'
            rec_mean = info.get('recommendationMean') # 1.0 - 5.0 typically

            if rec_key:
                sentiment = {
                    "key": rec_key,
                    "score": rec_mean
                }
        except Exception as e:
            pass

        return {
            "biggest_shareholder": biggest_holder,
            "sentiment": sentiment,
            "sector": t.info.get('sector'),
            "industry": t.info.get('industry'),
            "summary": t.info.get('longBusinessSummary')
        }

    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        return None


def generate_fun_stats(info):
    """Generates entertaining statistics and estimated rankings."""
    try:
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        market_cap = info.get('marketCap', 0)
        currency = info.get('currency', 'USD')

        # 1. Market Cap Ranking (Estimated)
        # yfinance reports JSE market caps in RAND, so ZAR/ZAc listings get
        # JSE-scale badges; the USD thresholds would wildly overrank them.
        rank_badge = "Unranked"
        if currency in ('ZAR', 'ZAc'):
            if market_cap > 500_000_000_000:  # R500B+
                rank_badge = "JSE Giant 🦁 (Top 10)"
            elif market_cap > 100_000_000_000:
                rank_badge = "JSE Top 40 Heavyweight 🏋️"
            elif market_cap > 20_000_000_000:
                rank_badge = "JSE Large Cap 🏢"
            elif market_cap > 5_000_000_000:
                rank_badge = "JSE Mid Cap 🚤"
            elif market_cap > 0:
                rank_badge = "JSE Small Cap 🧗"
        elif market_cap > 2000_000_000_000: # 2T (USD thresholds)
            rank_badge = "Top 5 Global 🌍"
        elif market_cap > 1000_000_000_000: # 1T
            rank_badge = "Top 10 Global 🏆"
        elif market_cap > 500_000_000_000: # 500B
            rank_badge = "Top 20 Global 🚀"
        elif market_cap > 200_000_000_000:
            rank_badge = "Blue Chip Titan 🏛️"
        elif market_cap > 50_000_000_000:
            rank_badge = "Large Cap Leader 🏢"
        elif market_cap > 10_000_000_000:
            rank_badge = "Mid Cap Mover 🚤"
        elif market_cap > 2_000_000_000:
            rank_badge = "Small Cap Challenger 🧗"
        elif market_cap > 0:
            rank_badge = "Micro Cap Gem 💎"

        # 2. Meal Index (Big Mac Index proxy)
        # USD stocks: Big Mac (~$5.69). JSE stocks are priced in ZAc (cents) or
        # ZAR, where the local staple benchmark is the KFC Streetwise 2 (~R45).
        meal_label = "Burger Index"
        meal_text = "N/A"
        if current_price and currency == 'USD':
            meals = current_price / 5.69
            if meals >= 1:
                meal_text = f"1 Share = {int(meals)} Big Macs 🍔"
            else:
                meal_text = f"1 Big Mac = {round(5.69 / current_price)} Shares 🍔"
        elif current_price and currency in ('ZAR', 'ZAc'):
            meal_label = "Streetwise Index"
            price_zar = current_price / 100 if currency == 'ZAc' else current_price
            meals = price_zar / 44.90
            if meals >= 1:
                meal_text = f"1 Share = {int(meals)} Streetwise 2s 🍗"
            else:
                meal_text = f"1 Streetwise 2 = {round(44.90 / price_zar)} Shares 🍗"

        # 3. Market Mood
        beta = info.get('beta', 1)
        mood = "Neutral 😐"
        if beta > 1.5:
             mood = "Wild Ride 🎢 (High Volatility)"
        elif beta > 1.1:
             mood = "Aggressive 🐂"
        elif beta < 0.8:
             mood = "Defensive 🛡️"
        elif beta < 0:
             mood = "Contrarian 🐻"

        return {
            "rank_badge": rank_badge,
            "burger_index": meal_text,
            "burger_label": meal_label,
            "market_mood": mood
        }
    except Exception as e:
        logger.error(f"Error generating fun stats: {e}")
        return {}


def get_key_stats(ticker: str):
    """
    Fetches fundamental statistics for the company.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # Helper to safely get value or formatting
        def fmt(key, is_pct=False, is_currency=False):
            val = info.get(key)
            if val is None: return None
            return val

        fun_stats = generate_fun_stats(info)

        stats = {
            "valuation": {
                "market_cap": fmt("marketCap"),
                "pe_ratio": fmt("trailingPE"),
                "forward_pe": fmt("forwardPE"),
                "peg_ratio": fmt("pegRatio"),
                "price_to_book": fmt("priceToBook"),
                "dividend_yield": normalize_dividend_yield(info.get("dividendYield")),
            },
            "financials": {
                "revenue": fmt("totalRevenue"),
                "revenue_growth": fmt("revenueGrowth"),
                "gross_margins": fmt("grossMargins"),
                "operating_margins": fmt("operatingMargins"),
                "profit_margins": fmt("profitMargins"),
                "ebitda": fmt("ebitda"),
            },
            "trading": {
                "beta": fmt("beta"),
                "short_ratio": fmt("shortRatio"),
                "target_high": fmt("targetHighPrice"),
                "target_low": fmt("targetLowPrice"),
                "target_mean": fmt("targetMeanPrice"),
                "recommendation_mean": fmt("recommendationMean"),
            },
            "insight": {
                "rank": fun_stats.get("rank_badge"),
                "burgers": fun_stats.get("burger_index"),
                "burgers_label": fun_stats.get("burger_label"),
                "mood": fun_stats.get("market_mood")
            }
        }
        return clean_data(stats)
    except Exception as e:
        logger.error(f"Error fetching stats for {ticker}: {e}")
        return None


def get_news(ticker: str):
    """
    Fetches latest news for the company using DuckDuckGo HTML search.
    """
    try:
        # Construct query
        query = f"{ticker} stock news"
        url = f"https://html.duckduckgo.com/html?q={urllib.parse.quote(query)}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        results = []

        # DDG HTML results are usually in .result__body
        for result in soup.select('.result'):
            try:
                title_tag = result.select_one('.result__a')
                if not title_tag: continue

                title = title_tag.get_text()
                link = title_tag.get('href')

                # DDG links are redirects, e.g. //duckduckgo.com/l/?uddg=...
                # We can try to extract 'uddg' param
                if 'duckduckgo.com/l/' in link:
                     parsed = urllib.parse.urlparse(link)
                     qs = urllib.parse.parse_qs(parsed.query)
                     if 'uddg' in qs:
                         link = qs['uddg'][0]

                snippet_tag = result.select_one('.result__snippet')
                snippet = snippet_tag.get_text() if snippet_tag else ""

                # Extract source from snippet or url
                domain = urllib.parse.urlparse(link).netloc.replace('www.', '')

                results.append({
                    "title": title,
                    "publisher": domain,
                    "link": link,
                    "date": "Recent", # DDG HTML doesn't reliably give dates
                    "thumbnail": None,
                    "summary": snippet
                })

                if len(results) >= 4: break
            except Exception:
                continue

        return results
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []


def is_safe_public_url(url: str) -> bool:
    """
    SSRF guard for server-side fetches. Returns True only for http/https URLs
    whose host resolves exclusively to public IP addresses. Blocks private,
    loopback, link-local (incl. cloud metadata 169.254.169.254), reserved,
    multicast and unspecified ranges, and any non-http(s) scheme.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    try:
        addr_infos = socket.getaddrinfo(host, None)
    except Exception:
        # Unresolvable host -> treat as unsafe.
        return False

    for info in addr_infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False

    return True


def get_article_content(url: str):
    """
    Fetches and extracts text content from a news URL.

    Hardened against SSRF: the initial URL and every redirect hop are validated
    against is_safe_public_url() before any request is made, and redirects are
    followed manually (allow_redirects=False) so an attacker cannot redirect from
    a public host to an internal one.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        current_url = url
        response = None
        for _ in range(5):  # cap redirect chain
            if not is_safe_public_url(current_url):
                logger.warning(f"Blocked unsafe URL (SSRF guard): {current_url}")
                return {"content": "This article could not be loaded for security reasons."}

            response = requests.get(current_url, headers=headers, timeout=10, allow_redirects=False)

            if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                if not location:
                    break
                # Resolve relative redirects against the current URL, then re-validate next loop.
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            break
        else:
            return {"content": "Could not load the article (too many redirects)."}

        if response is None:
            return {"content": "Could not load the article."}

        soup = BeautifulSoup(response.content, 'html.parser')

        # Heuristic to find main content
        # 1. Look for <article>
        article = soup.find('article')

        # 2. If no article, look for common main content divs
        if not article:
            for cls in ['main-content', 'story-content', 'article-body', 'post-content']:
                article = soup.find(class_=re.compile(cls))
                if article: break

        # 3. Fallback to extracting all paragraphs if reasonable count
        if not article:
            paras = soup.find_all('p')
            # Filter distinct content paragraphs (simple heuristic: length > 50 chars)
            content_paras = [p.get_text() for p in paras if len(p.get_text()) > 50]
            text = "\n\n".join(content_paras)
        else:
             # Extract text from found container
             text = ""
             for p in article.find_all(['p', 'h2', 'h3']):
                 text += p.get_text() + "\n\n"

        return {"content": text if len(text) > 100 else "Could not extract article content automatically. Please visit the link."}

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return {"content": f"Error loading article: {str(e)}"}


def get_calendar(ticker: str):
    """
    Fetches upcoming earnings and dividend events.
    """
    try:
        t = yf.Ticker(ticker)

        # Calendar returns a dict with keys like 'Dividend Date', 'Earnings Date', etc.
        cal = t.calendar

        events = []

        if cal:
            # Handle difference in return structure (sometimes dataframe, sometimes dict)
            # Recent yfinance returns simple dict

            # Earnings
            earnings_date = cal.get('Earnings Date')
            if earnings_date:
                # specific handling if it's a list
                if isinstance(earnings_date, list):
                     earnings_date = earnings_date[0]
                events.append({
                    "event": "Earnings Release",
                    "date": str(earnings_date).split(" ")[0]
                })

            # Dividends
            div_date = cal.get('Dividend Date')
            ex_div = cal.get('Ex-Dividend Date')

            if div_date:
                events.append({
                    "event": "Dividend Date",
                    "date": str(div_date).split(" ")[0]
                })
            if ex_div:
                events.append({
                    "event": "Ex-Dividend Date",
                    "date": str(ex_div).split(" ")[0]
                })

        return events
    except Exception as e:
        logger.error(f"Error fetching calendar for {ticker}: {e}")
        return []


def get_dividend_history(ticker: str, start_year: int = 2010):
    """
    Fetches historical dividend data.
    """
    try:
        t = yf.Ticker(ticker)
        dividends = t.dividends

        if dividends is None or dividends.empty:
            return None

        # Filter by date
        start_date = f"{start_year}-01-01"
        dividends = dividends[dividends.index >= start_date]

        if dividends.empty:
            return None

        # Group by year for annual growth ('YE' = year-end; 'Y' is deprecated in pandas 2.2)
        annual_div = dividends.resample('YE').sum()

        # Calculate Growth
        growth = annual_div.pct_change().dropna()

        div_data = []
        for date, val in dividends.items():
             div_data.append({
                 "date": str(date).split(" ")[0],
                 "value": val
             })

        annual_data = []
        for date, val in annual_div.items():
            annual_data.append({
                "year": date.year,
                "value": val,
                "growth": growth.get(date, 0)
            })

        return {
            "history": div_data,
            "annual": annual_data,
            "current_yield": normalize_dividend_yield(t.info.get('dividendYield')) or 0,
            "payout_ratio": t.info.get('payoutRatio', 0)
        }
    except Exception as e:
        logger.error(f"Error fetching dividends for {ticker}: {e}")
        return None


# Local JSE catalog so users can find SA stocks by company name without knowing the ticker.
_JSE_NAME_CATALOG = [
    ("ABG.JO", "Absa Group Ltd", ["absa"]),
    ("AGL.JO", "Anglo American", ["anglo american", "anglo"]),
    ("ANG.JO", "AngloGold Ashanti", ["anglogold", "anglo gold"]),
    ("ANH.JO", "Anheuser-Busch InBev", ["ab inbev", "inbev", "anheuser"]),
    ("APN.JO", "Aspen Pharmacare", ["aspen"]),
    ("BHG.JO", "BHP Group", ["bhp"]),
    ("BID.JO", "Bid Corp Ltd", ["bidcorp", "bid corp"]),
    ("BTI.JO", "British American Tobacco", ["bat", "british american"]),
    ("BVT.JO", "Bidvest Group", ["bidvest"]),
    ("CFR.JO", "Compagnie Financiere Richemont", ["richemont"]),
    ("CLS.JO", "Clicks Group", ["clicks"]),
    ("CPI.JO", "Capitec Bank", ["capitec"]),
    ("DSY.JO", "Discovery Ltd", ["discovery"]),
    ("EXX.JO", "Exxaro Resources", ["exxaro"]),
    ("FSR.JO", "FirstRand Ltd", ["firstrand", "first rand", "rmb"]),
    ("GFI.JO", "Gold Fields Ltd", ["gold fields", "goldfields"]),
    ("GLN.JO", "Glencore Plc", ["glencore"]),
    ("GRT.JO", "Growthpoint Properties", ["growthpoint"]),
    ("HAR.JO", "Harmony Gold Mining", ["harmony"]),
    ("IMP.JO", "Impala Platinum", ["implats", "impala"]),
    ("INL.JO", "Investec Ltd", ["investec"]),
    ("INP.JO", "Investec Plc", ["investec plc"]),
    ("MCG.JO", "MultiChoice Group", ["multichoice", "multi choice"]),
    ("MNP.JO", "Murray & Roberts", ["murray"]),
    ("MRP.JO", "Mr Price Group", ["mr price", "mrprice"]),
    ("MTN.JO", "MTN Group", ["mtn"]),
    ("NED.JO", "Nedbank Group", ["nedbank"]),
    ("NPH.JO", "Northam Platinum", ["northam"]),
    ("NPN.JO", "Naspers Ltd", ["naspers"]),
    ("NRP.JO", "NEPI Rockcastle", ["nepi", "rockcastle"]),
    ("OMU.JO", "Old Mutual Ltd", ["old mutual"]),
    ("OUT.JO", "OUTsurance Group", ["outsurance"]),
    ("PAN.JO", "Pan African Resources", ["pan african"]),
    ("PPH.JO", "Pepkor Holdings", ["pepkor", "pep"]),
    ("PRX.JO", "Prosus NV", ["prosus"]),
    ("REM.JO", "Remgro Ltd", ["remgro"]),
    ("RMI.JO", "Rand Merchant Investment", ["rmi"]),
    ("RNI.JO", "Reinet Investments", ["reinet"]),
    ("SBK.JO", "Standard Bank Group", ["standard bank", "stanbic"]),
    ("SHP.JO", "Shoprite Holdings", ["shoprite"]),
    ("SLM.JO", "Sanlam Ltd", ["sanlam"]),
    ("SOL.JO", "Sasol Ltd", ["sasol"]),
    ("SSW.JO", "Sibanye Stillwater", ["sibanye"]),
    ("VAL.JO", "Valterra Platinum", ["valterra"]),
    ("VOD.JO", "Vodacom Group", ["vodacom"]),
    ("WHL.JO", "Woolworths Holdings", ["woolworths", "woolies"]),
]

# Yahoo exchange / MIC-like codes → (ISO country, flag emoji, short venue label)
_EXCHANGE_COUNTRY = {
    "JNB": ("ZA", "🇿🇦", "JSE"),
    "JSE": ("ZA", "🇿🇦", "JSE"),
    "NMS": ("US", "🇺🇸", "NASDAQ"),
    "NGM": ("US", "🇺🇸", "NASDAQ"),
    "NCM": ("US", "🇺🇸", "NASDAQ"),
    "NAS": ("US", "🇺🇸", "NASDAQ"),
    "NYQ": ("US", "🇺🇸", "NYSE"),
    "NYSE": ("US", "🇺🇸", "NYSE"),
    "ASE": ("US", "🇺🇸", "NYSE"),
    "PCX": ("US", "🇺🇸", "NYSE"),
    "PNK": ("US", "🇺🇸", "OTC"),
    "OEM": ("US", "🇺🇸", "OTC"),
    "OQB": ("US", "🇺🇸", "OTC"),
    "OQX": ("US", "🇺🇸", "OTC"),
    "BTS": ("US", "🇺🇸", "CBOE"),
    "YHD": ("US", "🇺🇸", "US"),
    "LSE": ("GB", "🇬🇧", "LSE"),
    "LON": ("GB", "🇬🇧", "LSE"),
    "IOB": ("GB", "🇬🇧", "LSE"),
    "GER": ("DE", "🇩🇪", "XETRA"),
    "FRA": ("DE", "🇩🇪", "Frankfurt"),
    "MUN": ("DE", "🇩🇪", "Munich"),
    "STU": ("DE", "🇩🇪", "Stuttgart"),
    "BER": ("DE", "🇩🇪", "Berlin"),
    "HAM": ("DE", "🇩🇪", "Hamburg"),
    "DUS": ("DE", "🇩🇪", "Dusseldorf"),
    "EUX": ("DE", "🇩🇪", "Eurex"),
    "PAR": ("FR", "🇫🇷", "Euronext"),
    "EPA": ("FR", "🇫🇷", "Euronext"),
    "AMS": ("NL", "🇳🇱", "Euronext"),
    "AEX": ("NL", "🇳🇱", "Euronext"),
    "BRU": ("BE", "🇧🇪", "Euronext"),
    "LIS": ("PT", "🇵🇹", "Euronext"),
    "MIL": ("IT", "🇮🇹", "Borsa Italiana"),
    "MTA": ("IT", "🇮🇹", "Borsa Italiana"),
    "MAD": ("ES", "🇪🇸", "BME"),
    "MCE": ("ES", "🇪🇸", "BME"),
    "SWX": ("CH", "🇨🇭", "SIX"),
    "EBS": ("CH", "🇨🇭", "SIX"),
    "VIE": ("AT", "🇦🇹", "Vienna"),
    "OSL": ("NO", "🇳🇴", "Oslo"),
    "STO": ("SE", "🇸🇪", "Stockholm"),
    "CPH": ("DK", "🇩🇰", "Copenhagen"),
    "HEL": ("FI", "🇫🇮", "Helsinki"),
    "WSE": ("PL", "🇵🇱", "Warsaw"),
    "IST": ("TR", "🇹🇷", "Istanbul"),
    "TYO": ("JP", "🇯🇵", "TSE"),
    "JPX": ("JP", "🇯🇵", "TSE"),
    "OSA": ("JP", "🇯🇵", "OSE"),
    "HKG": ("HK", "🇭🇰", "HKEX"),
    "SHH": ("CN", "🇨🇳", "SSE"),
    "SHZ": ("CN", "🇨🇳", "SZSE"),
    "SSE": ("CN", "🇨🇳", "SSE"),
    "ASX": ("AU", "🇦🇺", "ASX"),
    "TOR": ("CA", "🇨🇦", "TSX"),
    "TSE": ("CA", "🇨🇦", "TSX"),
    "VAN": ("CA", "🇨🇦", "TSXV"),
    "CNQ": ("CA", "🇨🇦", "CSE"),
    "SAO": ("BR", "🇧🇷", "B3"),
    "BUE": ("AR", "🇦🇷", "BYMA"),
    "MEX": ("MX", "🇲🇽", "BMV"),
    "SET": ("TH", "🇹🇭", "SET"),
    "SES": ("SG", "🇸🇬", "SGX"),
    "KLS": ("MY", "🇲🇾", "Bursa"),
    "TWO": ("TW", "🇹🇼", "TWSE"),
    "TAI": ("TW", "🇹🇼", "TWSE"),
    "KOE": ("KR", "🇰🇷", "KRX"),
    "KSC": ("KR", "🇰🇷", "KRX"),
    "NSI": ("IN", "🇮🇳", "NSE"),
    "BSE": ("IN", "🇮🇳", "BSE"),
    "NZE": ("NZ", "🇳🇿", "NZX"),
}

# Yahoo / yfinance ticker suffix → country
_SUFFIX_COUNTRY = {
    "JO": ("ZA", "🇿🇦", "JSE"),
    "L": ("GB", "🇬🇧", "LSE"),
    "IL": ("GB", "🇬🇧", "LSE"),
    "DE": ("DE", "🇩🇪", "XETRA"),
    "F": ("DE", "🇩🇪", "Frankfurt"),
    "SG": ("DE", "🇩🇪", "Stuttgart"),
    "PA": ("FR", "🇫🇷", "Euronext"),
    "AS": ("NL", "🇳🇱", "Euronext"),
    "BR": ("BE", "🇧🇪", "Euronext"),
    "LS": ("PT", "🇵🇹", "Euronext"),
    "MI": ("IT", "🇮🇹", "Borsa Italiana"),
    "MC": ("ES", "🇪🇸", "BME"),
    "SW": ("CH", "🇨🇭", "SIX"),
    "VI": ("AT", "🇦🇹", "Vienna"),
    "OL": ("NO", "🇳🇴", "Oslo"),
    "ST": ("SE", "🇸🇪", "Stockholm"),
    "CO": ("DK", "🇩🇰", "Copenhagen"),
    "HE": ("FI", "🇫🇮", "Helsinki"),
    "WA": ("PL", "🇵🇱", "Warsaw"),
    "IS": ("TR", "🇹🇷", "Istanbul"),
    "T": ("JP", "🇯🇵", "TSE"),
    "HK": ("HK", "🇭🇰", "HKEX"),
    "SS": ("CN", "🇨🇳", "SSE"),
    "SZ": ("CN", "🇨🇳", "SZSE"),
    "AX": ("AU", "🇦🇺", "ASX"),
    "TO": ("CA", "🇨🇦", "TSX"),
    "V": ("CA", "🇨🇦", "TSXV"),
    "SA": ("BR", "🇧🇷", "B3"),
    "BA": ("AR", "🇦🇷", "BYMA"),
    "MX": ("MX", "🇲🇽", "BMV"),
    "BK": ("TH", "🇹🇭", "SET"),
    "SI": ("SG", "🇸🇬", "SGX"),
    "KL": ("MY", "🇲🇾", "Bursa"),
    "TW": ("TW", "🇹🇼", "TWSE"),
    "TWO": ("TW", "🇹🇼", "TWSE"),
    "KS": ("KR", "🇰🇷", "KRX"),
    "KQ": ("KR", "🇰🇷", "KOSDAQ"),
    "NS": ("IN", "🇮🇳", "NSE"),
    "BO": ("IN", "🇮🇳", "BSE"),
    "NZ": ("NZ", "🇳🇿", "NZX"),
}


def _listing_meta(symbol: str, exchange: str = ""):
    """Resolve country flag + venue label from exchange code or ticker suffix."""
    exch = (exchange or "").upper().strip()
    if exch in _EXCHANGE_COUNTRY:
        country, flag, venue = _EXCHANGE_COUNTRY[exch]
        return {"country": country, "flag": flag, "venue": venue}

    sym = (symbol or "").upper().strip()
    if "." in sym:
        suffix = sym.rsplit(".", 1)[-1]
        # Index symbols like ^J203.JO
        if suffix in _SUFFIX_COUNTRY:
            country, flag, venue = _SUFFIX_COUNTRY[suffix]
            return {"country": country, "flag": flag, "venue": venue}

    # Bare US tickers (no suffix) default to US
    if sym and not sym.startswith("^") and "." not in sym:
        return {"country": "US", "flag": "🇺🇸", "venue": exch or "US"}

    return {"country": "", "flag": "🌍", "venue": exch or "—"}


def _name_match_score(query: str, symbol: str, shortname: str, longname: str) -> float:
    """Higher = better match for company-name / ticker search."""
    q = (query or "").strip().lower()
    if not q:
        return 0.0

    sym = (symbol or "").lower()
    short = (shortname or "").lower()
    long = (longname or "").lower()
    base = sym.split(".")[0]

    score = 0.0
    if sym == q or base == q:
        score += 100
    elif sym.startswith(q) or base.startswith(q):
        score += 80
    elif q in sym:
        score += 40

    for name in (short, long):
        if not name:
            continue
        if name == q:
            score += 95
        elif name.startswith(q):
            score += 70
        elif q in name:
            score += 50
        else:
            # Token overlap (e.g. "standard bank" vs "Standard Bank Group Limited")
            q_tokens = [t for t in q.split() if len(t) > 1]
            if q_tokens and all(t in name for t in q_tokens):
                score += 65

    return score


def _is_south_african(result: dict) -> bool:
    """True when the listing is on the JSE / ZA market."""
    if (result.get("country") or "").upper() == "ZA":
        return True
    symbol = (result.get("symbol") or "").upper()
    if symbol.endswith(".JO"):
        return True
    exchange = (result.get("exchange") or "").upper()
    return exchange in {"JNB", "JSE"}


def _local_jse_matches(query: str):
    """Match local JSE catalog by ticker or company name aliases."""
    q = (query or "").strip().lower()
    if len(q) < 2:
        return []

    matches = []
    for symbol, name, aliases in _JSE_NAME_CATALOG:
        haystacks = [symbol.lower(), symbol.split(".")[0].lower(), name.lower(), *aliases]
        hit = False
        for h in haystacks:
            if q == h or q in h or h.startswith(q):
                hit = True
                break
            q_tokens = [t for t in q.split() if len(t) > 1]
            if q_tokens and all(t in h for t in q_tokens):
                hit = True
                break
        if hit:
            meta = _listing_meta(symbol, "JNB")
            matches.append({
                "symbol": symbol,
                "shortname": name,
                "longname": name,
                "exchange": "JNB",
                "exchDisp": "Johannesburg",
                "typeDisp": "Equity",
                "country": meta["country"],
                "flag": meta["flag"],
                "venue": meta["venue"],
                "_score": _name_match_score(q, symbol, name, name) + 25,  # boost local catalog
            })
    return matches


def search_tickers(query: str):
    """
    Search tickers by company name or symbol.
    Merges Yahoo Finance autocomplete with a local JSE name catalog, ranks
    South African listings first then by name relevance, and attaches country
    flag / venue (not city MIC codes).
    """
    q = (query or "").strip()
    if len(q) < 1:
        return []

    results_by_symbol = {}

    # 1) Local JSE name catalog (works even when Yahoo ranks ADRs first)
    for item in _local_jse_matches(q):
        results_by_symbol[item["symbol"]] = item

    # 2) Yahoo Finance autocomplete
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": q,
            "quotesCount": 20,
            "newsCount": 0,
            "enableFuzzyQuery": "true",
            "enableCb": "false",
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()
        quotes = data.get("quotes", [])

        preferred_types = {"EQUITY", "ETF", "INDEX", "MUTUALFUND", "FUND"}

        for quote in quotes:
            symbol = quote.get("symbol")
            if not symbol:
                continue

            quote_type = (quote.get("quoteType") or "").upper()
            type_disp = quote.get("typeDisp") or quote_type or ""
            if quote_type and quote_type not in preferred_types and type_disp.upper() not in {
                "EQUITY", "ETF", "INDEX", "FUND", "MUTUAL FUND"
            }:
                # Keep unknowns (Yahoo sometimes omits type); skip obvious junk
                if quote_type in {"OPTION", "FUTURE", "CURRENCY", "CRYPTOCURRENCY"}:
                    continue

            shortname = quote.get("shortname") or symbol
            longname = quote.get("longname") or ""
            exchange = quote.get("exchange") or ""
            meta = _listing_meta(symbol, exchange)
            score = _name_match_score(q, symbol, shortname, longname)

            existing = results_by_symbol.get(symbol)
            if existing and existing.get("_score", 0) >= score:
                # Keep local catalog entry but enrich missing fields
                if not existing.get("longname") and longname:
                    existing["longname"] = longname
                continue

            results_by_symbol[symbol] = {
                "symbol": symbol,
                "shortname": shortname,
                "longname": longname,
                "exchange": exchange,
                "exchDisp": quote.get("exchDisp") or meta["venue"],
                "typeDisp": type_disp,
                "country": meta["country"],
                "flag": meta["flag"],
                "venue": meta["venue"],
                "_score": score,
            }
    except Exception as e:
        logger.error(f"Error searching tickers for {query}: {e}")

    ranked = sorted(
        results_by_symbol.values(),
        key=lambda r: (
            0 if _is_south_african(r) else 1,
            -r.get("_score", 0),
            r.get("symbol", ""),
        ),
    )

    # Strip internal score before returning
    cleaned = []
    for r in ranked[:15]:
        item = {k: v for k, v in r.items() if not k.startswith("_")}
        cleaned.append(item)
    return cleaned


def get_financials(ticker: str):
    """
    Fetches financial data needed for Valuation (DCF).
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # 1. Free Cash Flow
        # yfinance often returns cashflow as a DataFrame in t.cashflow
        fcf = None
        try:
             cf = t.cashflow
             if cf is not None and not cf.empty:
                 # Look for 'Free Cash Flow' or calculate 'Total Cash From Operating Activities' - 'Capital Expenditures'
                 # Note: yfinance rows are localized, usually "Free Cash Flow" exists in recent versions
                 if "Free Cash Flow" in cf.index:
                     fcf = cf.loc["Free Cash Flow"].iloc[0] # Most recent
                 elif "Total Cash From Operating Activities" in cf.index and "Capital Expenditures" in cf.index:
                     fcf = cf.loc["Total Cash From Operating Activities"].iloc[0] + cf.loc["Capital Expenditures"].iloc[0] # CapEx is usually negative
        except Exception:
            pass

        # Fallback to info provided FCF if available (often not reliable/present)
        if fcf is None:
             fcf = info.get("freeCashflow")

        # 2. Shares Outstanding
        shares = info.get("sharesOutstanding")

        # 3. Beta
        beta = info.get("beta")

        # 4. WACC components
        suggested_discount_rate = core_math.calculate_wacc(ticker, beta)

        return {
            "fcf": fcf,
            "shares_outstanding": shares,
            "beta": beta,
            "price": info.get("currentPrice", info.get("regularMarketPreviousClose")),
            "suggested_discount_rate": suggested_discount_rate,
            "currency": info.get("currency", "USD")
        }
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker}: {e}")
        return None
