# Deployment Guide: Netlify + Backend + Neon Database

This guide will help you deploy your complete application stack:
- **Frontend** → Netlify (already done ✅)
- **Backend** → Render.com / Railway.app
- **Database** → Neon PostgreSQL

---

## Step 1: Setup Neon Database (5 minutes)

### 1.1 Create Neon Account & Project

1. Go to [https://console.neon.tech](https://console.neon.tech)
2. Sign up or log in
3. Click **"Create a project"**
   - Project name: `fin-fuz-sol`
   - Region: Choose closest to your users
   - PostgreSQL version: Latest (16+)
4. Click **"Create project"**

### 1.2 Get Database Connection String

1. In your Neon dashboard, click **"Connection Details"**
2. Copy the **connection string** (it looks like this):
   ```
   postgresql://username:password@ep-xxx-xxx.region.aws.neon.tech/dbname?sslmode=require
   ```
3. **Save this** - you'll need it for the backend deployment

---

## Step 2: Deploy Backend to Render.com (10 minutes)

> **Note**: If you prefer Railway.app or Fly.io, see Alternative Platforms section below.

### 2.1 Prerequisites

- Your code must be in a GitHub repository
- If not already on GitHub:
  ```bash
  cd /home/red/Public/Projects/fin_fuz_sol
  git add .
  git commit -m "Add production deployment configuration"
  git push origin main
  ```

### 2.2 Create Render Account

1. Go to [https://render.com](https://render.com)
2. Sign up with GitHub (easiest way)
3. Authorize Render to access your repositories

### 2.3 Deploy Backend

1. **Click "New +" → "Web Service"**

2. **Connect Repository**:
   - Find and select your `fin_fuz_sol` repository
   - Click "Connect"

3. **Configure Service**:
   - **Name**: `fin-fuz-sol-backend`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Choose Plan**: Select **"Free"**

5. **Add Environment Variables** (click "Advanced"):
   
   | Key | Value |
   |-----|-------|
   | `DATABASE_URL` | Paste your Neon connection string from Step 1.2 |
   | `SECRET_KEY` | Generate a random 32+ character string ([use this](https://randomkeygen.com/)) |
   | `PYTHON_VERSION` | `3.11` |

6. **Click "Create Web Service"**

7. **Wait for deployment** (5-10 minutes):
   - Watch the build logs
   - Once complete, you'll see: "Your service is live 🎉"
   - Copy your backend URL: `https://fin-fuz-sol-backend.onrender.com`

8. **Test Backend**:
   - Open: `https://YOUR-APP-NAME.onrender.com/`
   - You should see: `{"message": "Global Index Analyzer API is running"}`
   - If you see this, **backend is working!** ✅

---

## Step 3: Configure Netlify Frontend (2 minutes)

### 3.1 Add Environment Variable

1. Go to your Netlify dashboard
2. Select your site
3. Go to: **Site Settings → Environment Variables**
4. Click **"Add a variable"**

   | Key | Value |
   |-----|-------|
   | `VITE_API_BASE_URL` | `https://YOUR-BACKEND.onrender.com` |
   
   ⚠️ **Important**: 
   - Replace `YOUR-BACKEND` with your actual Render URL
   - **No trailing slash** at the end
   - Include `https://`

### 3.2 Redeploy Frontend

1. In Netlify, go to **Deploys** tab
2. Click **"Trigger deploy" → "Deploy site"**
3. Wait for deployment to complete (~1 minute)

### 3.3 Test Everything

1. Open your Netlify site URL
2. Open browser console (F12)
3. Look for: `✅ Using production API: https://your-backend.onrender.com`
4. Try registering a new user
5. Try logging in
6. Run an analysis

**If everything works, you're done! 🎉**

---

## Troubleshooting

### Frontend still trying to connect to localhost

**Check:**
1. Environment variable is set in Netlify dashboard
2. Variable name is exactly `VITE_API_BASE_URL` (case-sensitive)
3. You triggered a new deployment AFTER adding the variable
4. Open browser console and check what URL is being used

**Fix:**
```
# Vite only reads env vars during build time
# You MUST redeploy after adding environment variables
```

### Backend shows "Internal Server Error"

**Check Render logs:**
1. Go to Render dashboard → Your service
2. Click "Logs" tab
3. Look for error messages

**Common issues:**
- Database connection failed → Check `DATABASE_URL` format
- Missing dependencies → Check build logs

### Database connection errors

**Check:**
1. Neon connection string is correct
2. Connection string starts with `postgresql://` not `postgres://`
3. Connection string includes `?sslmode=require` at the end

**Test connection:**
```bash
# Install psycopg2 locally
pip install psycopg2-binary

# Test connection (replace with your URL)
python3 -c "import psycopg2; conn = psycopg2.connect('YOUR_DATABASE_URL'); print('✅ Connection successful')"
```

### CORS errors in browser

**Check:**
1. Backend is deployed and accessible
2. `allow_origins=["*"]` is in `main.py` (line 34)
3. Backend URL in Netlify matches your Render URL exactly

---

## Alternative Platforms

### Railway.app

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Add environment variables:
   - `DATABASE_URL`
   - `SECRET_KEY`
6. Set start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Deploy

### Fly.io

Requires Docker. Create `Dockerfile` in `backend/`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Then:
```bash
fly launch
fly secrets set DATABASE_URL="your-neon-url"
fly secrets set SECRET_KEY="your-secret-key"
fly deploy
```

---

## Environment Variables Summary

### Backend (Render/Railway/Fly)

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `DATABASE_URL` | ✅ Yes | `postgresql://user:pass@host.neon.tech/db?sslmode=require` | Neon database connection |
| `SECRET_KEY` | ✅ Yes | `supersecretkey123...` (32+ chars) | JWT token signing |
| `PYTHON_VERSION` | No | `3.11` | Python version (if needed by platform) |

### Frontend (Netlify)

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `VITE_API_BASE_URL` | ✅ Yes | `https://fin-fuz-sol-backend.onrender.com` | Backend API URL |

---

## Next Steps

After successful deployment:

1. **Test all features**:
   - User registration/login
   - Stock analysis
   - Premium features
   - Data export

2. **Monitor your apps**:
   - Render dashboard for backend logs
   - Netlify analytics for frontend
   - Neon dashboard for database stats

3. **Setup custom domain** (optional):
   - In Netlify: Add custom domain
   - In Render: Add custom domain
   - Update `VITE_API_BASE_URL` to use your custom backend domain

4. **Enable HTTPS** (automatic):
   - Both Netlify and Render provide free SSL certificates
   - Your sites will automatically use HTTPS

---

## Cost Estimate

- **Netlify**: Free tier (100GB bandwidth/month)
- **Render**: Free tier (limited hours, sleeps after inactivity)
- **Neon**: Free tier (3GB storage, 1 project)

**Total: $0/month** for low-traffic sites

**Upgrade when needed:**
- Render: $7/month (always-on, no sleep)
- Neon: $19/month (more storage, branches)

---

## Support

If you run into issues:

1. Check the Troubleshooting section above
2. Check Render logs for backend errors
3. Check browser console for frontend errors
4. Verify all environment variables are set correctly

