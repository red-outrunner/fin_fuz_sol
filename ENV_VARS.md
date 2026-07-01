# 🚀 Quick Start: Environment Variables Setup

## What You Need to Do

Your code is now ready for deployment! Here's what you need to configure:

---

## 1️⃣ Netlify Frontend (Site Settings → Environment Variables)

Add this single variable:

```
Variable Name: VITE_API_BASE_URL
Value: https://YOUR-BACKEND-URL.onrender.com
```

⚠️ **Important:** 
- Replace `YOUR-BACKEND-URL` with your actual backend URL (from Step 2)
- No trailing slash!
- Must include `https://`

**After adding, redeploy your site!**

---

## 2️⃣ Backend Hosting (Render/Railway/Fly.io)

Add these two variables:

### DATABASE_URL
```
Variable Name: DATABASE_URL
Value: postgresql://user:password@host.neon.tech/database?sslmode=require
```
📝 Get this from: [Neon Console](https://console.neon.tech) → Your Project → Connection Details

### SECRET_KEY
```
Variable Name: SECRET_KEY
Value: <any random 32+ character string>
```
🔐 Generate one at: [RandomKeyGen.com](https://randomkeygen.com/) (use "CodeIgniter Encryption Keys")

---

## 3️⃣ Optional / Advanced Backend Variables

All optional — the app runs with sensible defaults if these are unset.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Set to `production` to make a missing `SECRET_KEY` a hard startup failure. |
| `ALLOWED_ORIGINS` | local dev origins | Comma-separated list of frontend origins allowed by CORS, e.g. `https://your-app.netlify.app`. **Set this in production** or the deployed frontend's requests will be blocked. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `SENTRY_DSN` | _(unset)_ | Enables Sentry error tracking when set. No-op if unset or `sentry-sdk` isn't installed. Get the DSN from your [Sentry](https://sentry.io) project. |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.0` | Fraction of requests traced for performance monitoring (0.0–1.0). Only relevant when `SENTRY_DSN` is set. |
| `RATE_LIMIT` | `60/minute` | Per-IP request limit applied to all routes. |
| `RATE_LIMIT_BURST` | `1000/hour` | Secondary longer-window per-IP cap. |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | Set to `redis://...` for shared rate-limit state across multiple workers/instances. |
| `ADMIN_EMAIL` | `test@gmail.com` | Email of the auto-seeded admin account (full access to all tools). |
| `ADMIN_PASSWORD` | `12345678` | Password for the seeded admin. **Change both of these in production** — the defaults are for local/dev only. |
| `CACHE_DIR` | `cache_v3` | Directory for the parquet market-data cache. |
| `CACHE_TTL_SECONDS` | `86400` | How long (seconds) a cached price series is considered fresh. |

---

## Complete Deployment Guide

For detailed step-by-step instructions, see:
- **[DEPLOYMENT.md](file:///home/red/Public/Projects/fin_fuz_sol/DEPLOYMENT.md)** - Complete guide with screenshots

---

## Quick Links

1. **Neon Database**: https://console.neon.tech
2. **Render (Backend Host)**: https://render.com
3. **Netlify (Frontend Host)**: https://app.netlify.com

---

## Verification Checklist

After deploying, verify everything works:

- [ ] Neon database created ✅
- [ ] Backend deployed to Render/Railway ✅
- [ ] Backend health check works (visit `https://your-backend.onrender.com/`) ✅
- [ ] `DATABASE_URL` added to backend environment ✅
- [ ] `SECRET_KEY` added to backend environment ✅
- [ ] `VITE_API_BASE_URL` added to Netlify ✅
- [ ] Netlify site redeployed ✅
- [ ] Can register new user ✅
- [ ] Can login ✅
- [ ] Can run analysis ✅

---

## Need Help?

Check `DEPLOYMENT.md` for:
- Troubleshooting common issues
- Alternative hosting platforms
- Cost estimates
- Advanced configuration
