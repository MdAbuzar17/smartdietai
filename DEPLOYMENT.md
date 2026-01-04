# SmartDiet AI - Deployment Guide

## Overview
This guide helps you deploy the SmartDiet AI application to Render.com with PostgreSQL database support.

## Prerequisites
- Model files in `models/` folder:
  - `best_v8.pt` (YOLO model)
  - `meal_xgb_model.pkl` (XGBoost model)
  - `meal_label_encoder.pkl`
  - `x_feature_columns.pkl`
- CSV file: `dietdataset_new.csv`

## Local Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create .env File
Copy `.env.example` to `.env` and fill in your values:
```bash
copy .env.example .env
```

Edit `.env`:
```
DATABASE_URL=sqlite:///diet_recommendation.db
SPOONACULAR_API_KEY=your_actual_api_key_here
SECRET_KEY=your_random_secret_key_here
FLASK_ENV=development
PORT=8080
```

### 4. Run Locally
```bash
python app.py
```

Visit `http://localhost:8080`

## Render.com Deployment

### Step 1: Prepare Your Repository
1. Ensure all model files are in the `models/` folder
2. Commit your code to Git (DO NOT commit .env file!)
3. Push to GitHub/GitLab

### Step 2: Create PostgreSQL Database on Render
1. Go to Render Dashboard
2. Click "New +" → "PostgreSQL"
3. Name: `smartdiet-db`
4. Region: Choose closest to your users
5. Plan: Free or paid
6. Click "Create Database"
7. Copy the "External Database URL" - you'll need this

### Step 3: Create Web Service on Render
1. Click "New +" → "Web Service"
2. Connect your Git repository
3. Configure:
   - **Name**: `smartdiet-ai`
   - **Region**: Same as database
   - **Branch**: `main` (or your branch)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --timeout 120`

### Step 4: Set Environment Variables
In the "Environment" section, add:

| Key | Value |
|-----|-------|
| `DATABASE_URL` | (Paste the PostgreSQL External Database URL) |
| `SPOONACULAR_API_KEY` | Your Spoonacular API key |
| `SECRET_KEY` | Generate random string (use `python -c "import secrets; print(secrets.token_hex(32))"`) |
| `FLASK_ENV` | `production` |
| `PYTHON_VERSION` | `3.10.0` |

### Step 5: Deploy
1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Once deployed, you'll get a URL like `https://smartdiet-ai.onrender.com`

## Post-Deployment

### Create Database Schema
After first deployment, you need to create database tables. You can do this by:

1. Connect to your Render database using psql or a GUI tool
2. Run your schema creation SQL (you'll need to convert from SQLite to PostgreSQL)

**Note**: You may need to create a migration script to set up initial tables.

### Test Your Deployment
1. Visit your Render URL
2. Register a new account
3. Upload a food image
4. Test tracking and recommendations

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Ensure `requirements.txt` is correct
- Model files might be too large for free tier (consider using external storage)

### Database Connection Issues
- Ver ify DATABASE_URL is set correctly
- Check PostgreSQL database is running
- Ensure DATABASE_URL starts with `postgresql://` (Render may give `postgres://`)

### Model Loading Errors
- Ensure all `.pt` and `.pkl` files are in `models/` folder
- Check file paths are correct
- Model files might exceed memory limits (upgrade plan if needed)

### API Key Issues
- Verify SPOONACULAR_API_KEY is set in environment variables
- Check API key is valid and has remaining quota

## Performance Optimization

### For Production
1. Consider upgrading to paid tier for better performance
2. Enable CDN for static files
3. Use Redis for caching (optional)
4. Monitor logs regularly

### Memory Considerations
- YOLO and XGBoost models are memory-intensive
- Free tier: 512MB RAM (might be tight)
- Starter tier: 2GB RAM (recommended)

## Security Checklist
- ✅ API keys in environment variables
- ✅ Secret key is random and secure
- ✅ Debug mode is OFF in production
- ✅ .env file is in .gitignore
- ✅ Database credentials not hardcoded

## Mobile Responsiveness
The application is now fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablets (iPad, Android tablets)
- Mobile phones (iPhone, Android phones)

Test on multiple devices after deployment!

## Support
For issues:
1. Check Render logs
2. Verify environment variables
3. Test locally first
4. Check model files are accessible

## Maintenance
- Regularly update dependencies
- Monitor API usage (Spoonacular has limits)
- Back up PostgreSQL database
- Monitor application performance and errors
