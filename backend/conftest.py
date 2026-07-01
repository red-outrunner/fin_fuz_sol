"""Shared pytest config for the backend test suite.

Auth/tiers were removed, so endpoints are open and need no token override. We only
raise the rate limit so rapid test requests don't trip the limiter.
"""
import os

os.environ.setdefault("RATE_LIMIT", "100000/minute")
os.environ.setdefault("RATE_LIMIT_BURST", "100000/hour")
