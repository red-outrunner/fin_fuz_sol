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
    # Banks / financials
    "SBK.JO": {"name": "ICBC (Ind. & Comm. Bank of China)",   "percent": 0.196, "as_of": "2024"},
    "FSR.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.16,  "as_of": "2024"},
    "NED.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.147, "as_of": "2025"},
    "ABG.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.14,  "as_of": "2025"},
    "CPI.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.16,  "as_of": "2025"},
    "INL.JO": {"name": "Public Investment Corporation (PIC)", "percent": 0.17,  "as_of": "2025"},
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
    "VAL.JO": {"name": "Anglo American",                      "percent": 0.199, "as_of": "2025"},
    # Retail
    "SHP.JO": {"name": "Public Investment Corporation (PIC)", "percent": None,  "as_of": "2025"},
    # Rupert-family investment holdings
    "REM.JO": {"name": "Rupert family (Remgro control structure)", "percent": None,  "as_of": "2025"},
    "RNI.JO": {"name": "Anton Rupert Trust (Rupert family)",  "percent": 0.249, "as_of": "2025"},
    # NOTE: Remaining JSE Top 40 names return N/A on purpose — either widely-held
    # multinationals with no single dominant holder (Anglo American AGL, BHP BHG,
    # AB InBev ANH, British American Tobacco BTI, Glencore GLN, Mondi MNP, Investec
    # PLC INP) or ones a single largest shareholder couldn't be reliably sourced for
    # yet (GFI, EXX, NPH, WHL, MRP, CLS, APN, BVT, BID, GRT, NRP, RMH). Add them here
    # (with a source + as_of) as accurate data is confirmed rather than guessing.
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
        # Thresholds in Billions (USD approx)
        rank_badge = "Unranked"
        if market_cap > 2000_000_000_000: # 2T
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
        else:
            rank_badge = "Micro Cap Gem 💎"

        # 2. Burger Index (Big Mac Index proxy)
        # Approx Big Mac price: $5.69 (USD)
        burger_price = 5.69
        burgers = 0
        if current_price and currency == 'USD':
             burgers = current_price / burger_price

        burger_text = f"1 Share = {int(burgers)} Big Macs 🍔" if burgers > 0 else "N/A"

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
            "burger_index": burger_text,
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


def search_tickers(query: str):
    """
    Searches for tickers using Yahoo Finance Autocomplete API.
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": 10,
            "newsCount": 0,
            "enableFuzzyQuery": "true",
            "enableCb": "false"
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, params=params, headers=headers, timeout=5)
        data = response.json()

        quotes = data.get('quotes', [])

        results = []
        for quote in quotes:
            # Filter out non-equity if desired, currently keeping most valid types
            if 'symbol' not in quote: continue

            results.append({
                "symbol": quote['symbol'],
                "shortname": quote.get('shortname', quote['symbol']),
                "longname": quote.get('longname', ''),
                "exchange": quote.get('exchange', ''),
                "typeDisp": quote.get('typeDisp', '')
            })

        return results
    except Exception as e:
        logger.error(f"Error searching tickers for {query}: {e}")
        return []


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
