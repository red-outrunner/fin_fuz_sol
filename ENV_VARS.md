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
