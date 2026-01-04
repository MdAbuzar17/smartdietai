# ==================== IMPORTS ====================
import os
import re
import json
import time
import random
import base64
import hashlib
import pickle
import requests
import joblib
from decimal import Decimal
from datetime import date, datetime, timedelta
from pathlib import Path

# Data & Scientific
import numpy as np
import pandas as pd
import cv2

# Framework & Environment
from flask import Flask, redirect, url_for, render_template, request, flash, session

# Machine Learning
from ultralytics import YOLO

# Database - PostgreSQL Only
import psycopg2
from psycopg2.extras import RealDictCursor

import torch
import ultralytics.nn.tasks


# ==================== DATABASE CONNECTION & SETUP ====================
def get_connection():
    """Database connection for PostgreSQL"""
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        # Fallback for local development if not in .env
        # Ensure you have a local postgres DB named 'diet_recommendation'
        database_url = "postgresql://postgres:postgres@localhost:5432/diet_recommendation"
        
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
    try:
        conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise e

def create_tables():
    """Create tables if they don't exist (PostgreSQL)"""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS "user" (
            u_id SERIAL PRIMARY KEY,
            u_username VARCHAR(255),
            u_email VARCHAR(255) UNIQUE NOT NULL,
            u_password VARCHAR(255),
            u_age INTEGER,
            u_gender VARCHAR(50),
            u_vegan VARCHAR(50),
            u_allergy VARCHAR(255),
            u_weight FLOAT,
            u_feet INTEGER,
            u_inches INTEGER,
            u_bmi INTEGER,
            u_activitylevel VARCHAR(50),
            u_protein FLOAT,
            u_carb FLOAT,
            u_fat FLOAT,
            u_fiber FLOAT,
            u_calories FLOAT,
            u_journey INTEGER,
            u_bodyfat FLOAT,
            u_status VARCHAR(50),
            u_startdate VARCHAR(50) DEFAULT TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD')
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tracking (
            track_id SERIAL PRIMARY KEY,
            track_date VARCHAR(50),
            u_id INTEGER REFERENCES "user"(u_id),
            track_calorie FLOAT,
            track_protein FLOAT,
            track_fat FLOAT,
            track_carb FLOAT,
            track_breakfast TEXT DEFAULT '',
            track_lunch TEXT DEFAULT '',
            track_snack TEXT DEFAULT '',
            track_dinner TEXT DEFAULT ''
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS progress (
            p_id SERIAL PRIMARY KEY,
            u_id INTEGER REFERENCES "user"(u_id),
            p_date VARCHAR(50),
            p_weight FLOAT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS progress_week (
            pw_id SERIAL PRIMARY KEY,
            u_id INTEGER REFERENCES "user"(u_id),
            pw_num INTEGER,
            pw_weight FLOAT
        )
        """
    )
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        conn.commit()
        cur.close()
        print("Database tables created successfully (if not existed).")
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        if conn is not None:
            conn.close()

# Initialize Database
create_tables()

# ==================== FLASK APP CONFIGURATION ====================
app = Flask(__name__)
# Use environment variable for secret key, fallback to default for local dev
app.secret_key = os.environ.get('SECRET_KEY', 'honsproject-dev-key-change-in-production')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.globals.update(zip=zip)

bf_meal = []
onefood = []
lunch_meal = []
dinner_meal = []
snack_meal = []

# app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# Duplicate imports removed




# ==================== API CONFIGURATION ====================
# Load API key from environment variable for security
SPOONACULAR_API_KEY = os.environ.get('SPOONACULAR_API_KEY', '8d4e880bc97d4edc9a17b4a6d46e69ec')
SPOON_URL = "https://api.spoonacular.com/recipes/parseIngredients"

_cache = {}

def call_spoonacular(query):
    # Fallback for known Indian foods if API returns 0/low calories
    # Values are approximate standards per piece or per 100g
    FALLBACK_NUTRITION = {
        # Classes from Model
        "aloogobi": {"calories": 150, "protein": 3, "fat": 7, "carbs": 18, "fiber": 4, "sugar": 3, "serving_weight_g": 100},
        "aloomasala": {"calories": 140, "protein": 2, "fat": 6, "carbs": 20, "fiber": 3, "sugar": 2, "serving_weight_g": 100},
        "bhatura": {"calories": 290, "protein": 7, "fat": 12, "carbs": 38, "fiber": 2, "sugar": 2, "serving_weight_g": 90},
        "bhindimasala": {"calories": 80, "protein": 2, "fat": 5, "carbs": 8, "fiber": 3, "sugar": 3, "serving_weight_g": 100},
        "biryani": {"calories": 200, "protein": 8, "fat": 8, "carbs": 25, "fiber": 2, "sugar": 1, "serving_weight_g": 100},
        "chai": {"calories": 100, "protein": 3, "fat": 3, "carbs": 15, "fiber": 0, "sugar": 12, "serving_weight_g": 150},
        "chole": {"calories": 223, "protein": 10, "fat": 10, "carbs": 24, "fiber": 8, "sugar": 3, "serving_weight_g": 100},
        "coconutchutney": {"calories": 150, "protein": 2, "fat": 14, "carbs": 3, "fiber": 4, "sugar": 2, "serving_weight_g": 50},
        "dal": {"calories": 120, "protein": 6, "fat": 3, "carbs": 18, "fiber": 4, "sugar": 1, "serving_weight_g": 100},
        "dosa": {"calories": 168, "protein": 3.9, "fat": 3.7, "carbs": 29, "fiber": 0.9, "sugar": 0, "serving_weight_g": 80},
        "dumaloo": {"calories": 170, "protein": 3, "fat": 10, "carbs": 18, "fiber": 3, "sugar": 4, "serving_weight_g": 100},
        "fishcurry": {"calories": 140, "protein": 15, "fat": 8, "carbs": 5, "fiber": 1, "sugar": 1, "serving_weight_g": 100},
        "ghevar": {"calories": 400, "protein": 4, "fat": 20, "carbs": 50, "fiber": 1, "sugar": 30, "serving_weight_g": 100},
        "greenchutney": {"calories": 30, "protein": 1, "fat": 0, "carbs": 6, "fiber": 2, "sugar": 2, "serving_weight_g": 30},
        "gulabjamun": {"calories": 150, "protein": 2, "fat": 6, "carbs": 22, "fiber": 0, "sugar": 18, "serving_weight_g": 50},
        "idli": {"calories": 39, "protein": 2, "fat": 0.2, "carbs": 8, "fiber": 0, "sugar": 0, "serving_weight_g": 40},
        "jalebi": {"calories": 150, "protein": 1, "fat": 5, "carbs": 25, "fiber": 0, "sugar": 20, "serving_weight_g": 50},
        "kabab": {"calories": 150, "protein": 12, "fat": 10, "carbs": 4, "fiber": 1, "sugar": 1, "serving_weight_g": 60},
        "kheer": {"calories": 200, "protein": 5, "fat": 8, "carbs": 28, "fiber": 1, "sugar": 18, "serving_weight_g": 100},
        "kulfi": {"calories": 200, "protein": 5, "fat": 10, "carbs": 22, "fiber": 0, "sugar": 18, "serving_weight_g": 80},
        "lassi": {"calories": 150, "protein": 6, "fat": 6, "carbs": 18, "fiber": 0, "sugar": 15, "serving_weight_g": 200},
        "muttoncurry": {"calories": 250, "protein": 18, "fat": 18, "carbs": 6, "fiber": 2, "sugar": 1, "serving_weight_g": 100},
        "onionpakoda": {"calories": 315, "protein": 8, "fat": 18, "carbs": 30, "fiber": 4, "sugar": 3, "serving_weight_g": 100},
        "palakpaneer": {"calories": 180, "protein": 9, "fat": 14, "carbs": 6, "fiber": 3, "sugar": 2, "serving_weight_g": 100},
        "poha": {"calories": 180, "protein": 3, "fat": 6, "carbs": 28, "fiber": 2, "sugar": 2, "serving_weight_g": 100},
        "rajmacurry": {"calories": 140, "protein": 6, "fat": 5, "carbs": 18, "fiber": 6, "sugar": 2, "serving_weight_g": 100},
        "rasmalai": {"calories": 180, "protein": 6, "fat": 8, "carbs": 20, "fiber": 0, "sugar": 14, "serving_weight_g": 80},
        "samosa": {"calories": 260, "protein": 6, "fat": 17, "carbs": 24, "fiber": 2, "sugar": 2, "serving_weight_g": 80},
        "shahipaneer": {"calories": 300, "protein": 10, "fat": 25, "carbs": 10, "fiber": 2, "sugar": 4, "serving_weight_g": 100},
        "whiterice": {"calories": 130, "protein": 2, "fat": 0, "carbs": 28, "fiber": 0, "sugar": 0, "serving_weight_g": 100},
        
        # Common Variations/Aliases
        "chole bhature": {"calories": 450, "protein": 12, "fat": 20, "carbs": 55, "fiber": 8, "sugar": 5, "serving_weight_g": 200}, # Combo
        "masala dosa": {"calories": 387, "protein": 5, "fat": 15, "carbs": 55, "fiber": 2, "sugar": 1, "serving_weight_g": 180},
        "sambar": {"calories": 60, "protein": 2, "fat": 1.5, "carbs": 9, "fiber": 2, "sugar": 2, "serving_weight_g": 100},
        "samba": {"calories": 60, "protein": 2, "fat": 1.5, "carbs": 9, "fiber": 2, "sugar": 2, "serving_weight_g": 100},
        "vada": {"calories": 97, "protein": 2, "fat": 7, "carbs": 6, "fiber": 1, "sugar": 0, "serving_weight_g": 35},
        "chapati": {"calories": 104, "protein": 3, "fat": 3, "carbs": 15, "fiber": 3, "sugar": 0, "serving_weight_g": 40},
        "puri": {"calories": 101, "protein": 1, "fat": 6, "carbs": 11, "fiber": 0.5, "sugar": 0, "serving_weight_g": 30},
        "chicken biryani": {"calories": 220, "protein": 10, "fat": 10, "carbs": 22, "fiber": 2, "sugar": 0, "serving_weight_g": 100},
        "jamun": {"calories": 150, "protein": 2, "fat": 6, "carbs": 22, "fiber": 0, "sugar": 18, "serving_weight_g": 50}, 
        "rajma": {"calories": 140, "protein": 6, "fat": 5, "carbs": 18, "fiber": 6, "sugar": 2, "serving_weight_g": 100}, 
    }

    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "includeNutrition": "true"
    }
    data_payload = {
        "ingredientList": query,
        "servings": 1
    }
    
    info = None
    item = None
    
    # Attempt Spoonacular API Call
    try:
        r = requests.post(SPOON_URL, params=params, data=data_payload, timeout=10)
        if r.status_code == 200:
            json_data = r.json()
            if json_data:
                item = json_data[0]
                nutrients = item.get("nutrition", {}).get("nutrients", [])
                
                def get_amount(name):
                    for n in nutrients:
                        if n["name"].lower() == name.lower():
                            return n["amount"]
                    return 0.0
                
                info = {
                    "food": item.get("originalName") or item.get("name"),
                    "serving_weight_g": item.get("amount") if item.get("unitShort") in ["g", "grams"] else 100,
                    "calories": get_amount("Calories"),
                    "protein": get_amount("Protein"),
                    "carbs": get_amount("Carbohydrates"),
                    "sugars": get_amount("Sugar"),
                    "fiber": get_amount("Fiber"),
                    "fat": get_amount("Fat"),
                    "unit": item.get("unit")
                }
    except:
        pass # Fallback will pick it up

    # If no info or 0 calories, try fallback
    has_data = info and info.get("calories", 0) > 1.0
    
    if not has_data:
        # Determine search text for fallback
        if info:
             search_text = (info.get("food", "") + " " + query.lower()).strip()
        else:
             search_text = query.lower().strip()
             
        for k, v in FALLBACK_NUTRITION.items():
            if k in search_text:
                # Local Parsing for Quantity and Size
                # 1. Parse Quantity (leading number)
                quantity_match = re.search(r'^(\d+(\.\d+)?)', query.strip())
                amount = float(quantity_match.group(1)) if quantity_match else 1.0

                # 2. Parse Size (small/medium/large)
                size_multiplier = 1.0
                if "small" in search_text:
                    size_multiplier = 0.75
                elif "large" in search_text:
                    size_multiplier = 1.25 # Conservative large multiplier
                elif "medium" in search_text:
                    size_multiplier = 1.0 
                    
                # If the query provided an explicit amount that contradicts the API (or API is empty)
                # We trust our local parse for fallback items more than "1.0" default
                
                # Check if we should use weight-based or unit-based scaling
                is_weight = False
                # If user query has "g" or "grams" explicitly, treat as weight
                if re.search(r'\d+\s*(g|gram)', query.lower()):
                    is_weight = True
                    # Re-parse amount if it was like "100g"
                    weight_match = re.search(r'(\d+(\.\d+)?)\s*(g|gram)', query.lower())
                    if weight_match:
                        amount = float(weight_match.group(1))
                elif item and item.get("unitShort") in ["g", "grams"]:
                    is_weight = True
                    amount = item.get("amount", amount) # Trust API if it parsed grams successfully
                
                # Calculate
                new_info = info if info else {}
                new_info["food"] = k.title() # Use fallback name
                new_info["unit"] = item.get("unit") if item else "serving"
                
                if is_weight:
                   # Standardize logic
                    per_g_cal = v["calories"] / v["serving_weight_g"]
                    per_g_prot = v["protein"] / v["serving_weight_g"]
                    per_g_fat = v["fat"] / v["serving_weight_g"]
                    per_g_carb = v["carbs"] / v["serving_weight_g"]
                    
                    new_info["calories"] = per_g_cal * amount
                    new_info["protein"] = per_g_prot * amount
                    new_info["fat"] = per_g_fat * amount
                    new_info["carbs"] = per_g_carb * amount
                    new_info["serving_weight_g"] = amount
                    
                    # Size doesn't apply to explicit weight (100g small idli is still 100g)
                else:
                    # Unit based logic
                    final_multiplier = amount * size_multiplier
                    
                    new_info["calories"] = v["calories"] * final_multiplier
                    new_info["protein"] = v["protein"] * final_multiplier
                    new_info["fat"] = v["fat"] * final_multiplier
                    new_info["carbs"] = v["carbs"] * final_multiplier
                    new_info["serving_weight_g"] = v["serving_weight_g"] * final_multiplier
                
                # Zero out others if not in default info
                if "sugars" not in new_info: new_info["sugars"] = 0
                if "fiber" not in new_info: new_info["fiber"] = 0
                
                return new_info

    return info
_cache = {}

# ==================== MODEL LOADING (once at startup) ====================
# Use relative paths for deployment compatibility
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_v8.pt"

# Load YOLO model
print(f"Loading YOLO model from: {MODEL_PATH}")

torch.serialization.add_safe_globals(
    [ultralytics.nn.tasks.DetectionModel]
)

model = YOLO(str(MODEL_PATH))
class_names = model.model.names
print(f"YOLO model loaded successfully with {len(class_names)} classes")

def clean_class_name(name):
    spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    return spaced.lower().strip()


def get_nutrition_info(food_name):
    key = food_name.lower().strip()
    if key in _cache:
        return _cache[key]

    payload = food_name
    
    info = call_spoonacular(payload)
    if info:
        _cache[key] = info
        return info

    return None
DEFAULT_GRAMS_PER_CM2 = 0.8
DEFAULT_CAL_PER_100G = 150


def estimate_px_per_cm(img_width, plate_cm=25.0):
    return (img_width * 0.90) / plate_cm


def estimate_calories(area_px, px_per_cm, class_name):
    area_cm2 = area_px / (px_per_cm ** 2)
    grams = area_cm2 * DEFAULT_GRAMS_PER_CM2

    qname = clean_class_name(class_name)
    nutrition = get_nutrition_info(qname)

    if nutrition and nutrition["calories"] and nutrition["serving_weight_g"]:
        cal_per_100g = (nutrition["calories"] / nutrition["serving_weight_g"]) * 100
    else:
        cal_per_100g = DEFAULT_CAL_PER_100G

    kcal = grams * (cal_per_100g / 100)

    return kcal, grams, area_cm2, nutrition, cal_per_100g
DEFAULT_GRAMS_PER_CM2 = 0.8
DEFAULT_CAL_PER_100G = 150


def estimate_px_per_cm(img_width, plate_cm=25.0):
    return (img_width * 0.90) / plate_cm


def estimate_calories(area_px, px_per_cm, class_name):
    area_cm2 = area_px / (px_per_cm ** 2)
    grams = area_cm2 * DEFAULT_GRAMS_PER_CM2

    qname = clean_class_name(class_name)
    nutrition = get_nutrition_info(qname)

    if nutrition and nutrition["calories"] and nutrition["serving_weight_g"]:
        cal_per_100g = (nutrition["calories"] / nutrition["serving_weight_g"]) * 100
    else:
        cal_per_100g = DEFAULT_CAL_PER_100G

    kcal = grams * (cal_per_100g / 100)

    return kcal, grams, area_cm2, nutrition, cal_per_100g
def detect_and_visualize(img):
    h, w = img.shape[:2]

    px_per_cm = estimate_px_per_cm(w)

    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]

    # --- Annotated output image ---
    # results.plot() already returns a BGR image (OpenCV style)
    output = results.plot()               # BGR
    # DO NOT convert to RGB here for Flask/web

    detections = []
    items_with_calories = []
    total_kcal = 0

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names.get(cls_id, "Unknown")

        area_px = max(1, (x2 - x1) * (y2 - y1))

        kcal, grams, area_cm2, nutrition, cal_per_100g = estimate_calories(
            area_px, px_per_cm, class_name
        )

        total_kcal += kcal

        detections.append({
            "class": class_name,
            "confidence": conf,
            "area_px": area_px,
            "area_cm2": area_cm2,
            "grams_estimated": grams,
            "calories_estimated": kcal,
            "cal_per_100g_used": cal_per_100g,
            "nutritionix": nutrition
        })

        items_with_calories.append({
            "label": class_name,
            "calories": round(kcal, 1)
        })

    # IMPORTANT: encode BGR image directly
    _, buffer = cv2.imencode(".jpg", output)
    result_bytes = buffer.tobytes()

    return result_bytes, total_kcal, items_with_calories, detections

@app.route('/')
@app.route('/first')
def first():
    return render_template('home.html')


@app.route('/predict')
def index1():
    return render_template('index1.html')

@app.route('/about1')
def about1():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- BLOCK C (route) ----------------------
# Replace your existing /prediction1 route with this corrected version
@app.route('/prediction1', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index1.html', error="No file selected")

    file = request.files['file']

    if file.filename == '':
        return render_template('index1.html', error="No file selected")

    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result_bytes, total_calories, items_with_calories, detections = detect_and_visualize(img)

    encoded = base64.b64encode(result_bytes).decode()

    return render_template(
        'index1.html',
        filename=f"data:image/jpeg;base64,{encoded}",
        total_calories=round(total_calories, 1),
        items_with_calories=items_with_calories,
        detections=detections,
        name=file.filename,
        error=None
    )


@app.route("/delete_food", methods =['GET','POST'])
def delete_food():
	if request.method == 'GET':
		return redirect(url_for('add_successful'))

	else:
		mealtime = request.form['mealtime']
		item = request.form['fname']

		uid = session['uid']
		track_date = str(datetime.today().strftime ('%Y-%m-%d'))
		
		bf_list = []
		lunch_list = []
		dinner_list = []
		snack_list = []
		# for i in session['bf_numbers']:

			
		# 	print(float(i))
		# 	session.modified = True
		# 	print(type(i))
		if mealtime == "Breakfast":
			
			for i in bf_meal:
				if item == i[0]:
					bf_meal.remove(i)

			for i in session['bf_meal']:
				if item == i[0]:
					session['bf_meal'].remove(i)
					session.modified = True

			
					

				
					for j in session['bf_numbers']:
						j = round(Decimal(j), 2)
						bf_list.append(j)
					
				
					bf_list[0]-=round(Decimal(i[3]), 2)
					bf_list[1]-=round(Decimal(i[4]), 2)
					bf_list[2]-=round(Decimal(i[5]), 2)
					bf_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[2].split(",")[:-1]
							item_to_db = ""

							if item in item_in_db:
								item_in_db.remove(item)

							item_to_db = ",".join(item_in_db)+","

							cur2.execute("update tracking set track_breakfast=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except Exception as e:
						return (f'{e}')
					finally:
						conn.close()


					session['bf_numbers'].clear()

					[x for x in session['bf_numbers'] if x]

					session.modified = True
					for i in bf_list:
						session['bf_numbers'].append(i)	
						session.modified = True

					

					
					
					

		
		if mealtime == "Lunch":
			foodlist = session['lunch_meal']

			for i in lunch_meal:
				if item == i[0]:
					lunch_meal.remove(i)

			for i in foodlist:
				if item == i[0]:
					foodlist.remove(i)
				
			
					for j in session['lunch_numbers']:
						j = round(Decimal(j), 2)
						lunch_list.append(j)
					
				
					lunch_list[0]-=round(Decimal(i[3]), 2)
					lunch_list[1]-=round(Decimal(i[4]), 2)
					lunch_list[2]-=round(Decimal(i[5]), 2)
					lunch_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[3].split(",")[:-1]
							item_to_db = ""

							if item in item_in_db:
								item_in_db.remove(item)

							item_to_db = ",".join(item_in_db)+","
							
							cur2.execute("update tracking set track_lunch=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except Exception as e:
						return (f'{e}')
					finally:
						conn.close()

					session['lunch_numbers'].clear()

					[x for x in session['lunch_numbers'] if x]

					session.modified = True
					for i in lunch_list:
						session['lunch_numbers'].append(i)	
						session.modified = True


		if mealtime == "Snack":
			foodlist = session['snack_meal']

			for i in snack_meal:
				if item == i[0]:
					snack_meal.remove(i)

			for i in foodlist:
				if item == i[0]:
					foodlist.remove(i)
				
			
					for j in session['snack_numbers']:
						j = round(Decimal(j), 2)
						snack_list.append(j)
					
				
					snack_list[0]-=round(Decimal(i[3]), 2)
					snack_list[1]-=round(Decimal(i[4]), 2)
					snack_list[2]-=round(Decimal(i[5]), 2)
					snack_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[4].split(",")[:-1]
							item_to_db = ""

							if item in item_in_db:
								item_in_db.remove(item)

							item_to_db = ",".join(item_in_db)+","
				
							cur2.execute("update tracking set track_snack=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except Exception as e:
						return (f'{e}')
					finally:
						conn.close()

					session['snack_numbers'].clear()

					[x for x in session['snack_numbers'] if x]

					session.modified = True
					for i in snack_list:
						session['snack_numbers'].append(i)	
						session.modified = True

					

		if mealtime == "Dinner":
			foodlist = session['dinner_meal']
			for i in dinner_meal:
				if item == i[0]:
					dinner_meal.remove(i)

			for i in foodlist:
				if item == i[0]:
					foodlist.remove(i)
				
			
					for j in session['dinner_numbers']:
						j = round(Decimal(j), 2)
						dinner_list.append(j)
					
				
					dinner_list[0]-=round(Decimal(i[3]), 2)
					dinner_list[1]-=round(Decimal(i[4]), 2)
					dinner_list[2]-=round(Decimal(i[5]), 2)
					dinner_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[5].split(",")[:-1]
							item_to_db = ""

							if item in item_in_db:
								item_in_db.remove(item)

							item_to_db = ",".join(item_in_db)+","
							cur2.execute("update tracking set track_dinner=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except Exception as e:
						return (f'{e}')
					finally:
						conn.close()

					session['dinner_numbers'].clear()

					[x for x in session['dinner_numbers'] if x]

					session.modified = True
					for i in dinner_list:
						session['dinner_numbers'].append(i)	
						session.modified = True

					
				
								
		return render_template('add_food.html',mealtime=mealtime)
	


# Old get_connection removed


def pwd_security(passwd):
	"""A strong password must be at least 8 characters long
	   and must contain a lower case letter, an upper case letter,
	   and at least 3 digits.
	   Returns True if passwd meets these criteria, otherwise returns False.
	   """
	# check password length
	# check password for uppercase, lowercase and numeric chars
	hasupper = False	
	haslower = False
	digitcount = 0
	digit= False
	strong = False
	length = True
	special = False
	for c in passwd:
		if (c.isupper()==True):
			hasupper= True
		elif (c.islower()==True):
			haslower=True
		elif (c.isdigit()==True):
			digitcount+=1
			digit = True
		elif re.findall('[^A-Za-z0-9]',c):
			special = True
	if hasupper == True and haslower == True and digit == True and special == True:
		strong = True
	if len(passwd) <8:
		length = False
	return strong,haslower,hasupper,digit,length, special

def pwd_encode(pwd):
	secure_pwd =hashlib.md5(pwd.encode()).hexdigest()
	return secure_pwd



@app.route("/update_profile", methods =['GET','POST'] )
def edit_weight():
	if request.method == 'GET':

		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_id=%s",(session['uid'],))
				u_data = cur.fetchone()
				name = u_data[1]
				age = u_data[4]
				weight = u_data[6]
				email = u_data[5]
				password = session['u_pass']
				gender = u_data[3]
				ft = u_data[7]
				inch = u_data[8]
				vegan = u_data[12]
				allergy = u_data[13]

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		

		return render_template('edit_profile.html', name= name,
													age = age,
													weight = weight,
													email = email,
													password = password,
													gender = gender,
													ft = ft,
													inch = inch,
													vegan = vegan,
													allergy = allergy)
	else:
		name = request.form['name']
		age = int(request.form['age'])
		weight_kg = int(request.form['weight'])
		email = request.form['email']
		password = session['u_pass'] # Keep existing password
		gender = request.form['gender']
		feet = int(request.form['feet'])
		inches = int(request.form['inches'])
		vegan = request.form['vegan']
		allergy = request.form['allergy']
		activity_level = request.form['activity']

		height_bmi = int((feet * 12) + inches)
		height_m = height_bmi * 0.0254
		
		# Metric BMI: kg / m^2
		BMI = weight_kg / (height_m * height_m)

		bmr = 0
		bodyfat = 0
		
		if gender == "male":
			# Mifflin-St Jeor: 10*kg + 6.25*cm - 5*age + 5
			bmr = int((10 * weight_kg) + (15.88 * height_bmi) - (5 * age) + 5)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 16.2)
		else:
			bmr = int((10 * weight_kg) + (15.88 * height_bmi) - (5 * age) - 161)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 5.4)

		calorie = 0

		if activity_level == "sedentary":
			calorie = int(bmr*1.2)

		elif activity_level == "lightly active":
			calorie = int(bmr * 1.375)

		elif activity_level == "moderately active":
			calorie = int(bmr * 1.55)

		elif activity_level == "very active":
			calorie = int(bmr * 1.725)

		elif activity_level == "extra active":
			calorie = int(bmr * 1.9)

		body_status = ""
		if BMI < 18.5:
			body_status = "underweight"

		elif BMI >= 18.5 and BMI <= 24.9 :
			body_status = "healthy weight"

		elif BMI >= 25 and BMI <= 29.9 :
			body_status = "overweight"

		elif BMI >= 30 :
			body_status = "obese"

		protein = int(((calorie-500) * 0.30)/4)
		carb = int(((calorie-500)* 0.40)/4)
		fat = int(((calorie-500) * 0.30)/9)
		fiber = int(calorie/1000*14)

		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("update user set u_username=%s, u_email=%s, u_age=%s, u_gender=%s, u_vegan=%s, u_allergy=%s, u_weight=%s, u_feet=%s, u_inches=%s, u_bmi=%s, u_activitylevel=%s, u_protein=%s, u_carb=%s, u_fat=%s, u_fiber=%s, u_calories=%s, u_bodyfat=%s, u_status=%s where u_id=%s", (name, email, age, gender, vegan, allergy, weight_kg, feet, inches, int(BMI), activity_level, protein, carb, fat, fiber, calorie, bodyfat, body_status, session['uid']))
				conn.commit()
				
				# Update session info
				session['u_info'] = [] # Clear and reload
				cur.execute("select * from user where u_id=%s",(session['uid'],))
				u_info = cur.fetchone()
				for row in u_info:
					session['u_info'].append(row)

				# SYNC TO DAILY PROGRESS (User Request)
				# Also update 'progress' table for today so it shows in daily/weekly stats
				cur.execute("update progress set p_weight=%s where p_date=%s and u_id=%s",(weight_kg,datetime.today().strftime ('%Y-%m-%d'),session['uid'],))
				if cur.rowcount == 0:
					cur.execute("insert into progress (u_id,p_date,p_weight) values (%s,%s,%s)", (session['uid'], datetime.today().strftime('%Y-%m-%d'), weight_kg))
				
				# Update Weekly Average Logic (duplicated from index route)
				days = []
				track_date = datetime.today().strftime ('%Y-%m-%d')
				# Handle potential None for startdate, accessing safely
				if len(session['u_info']) > 0 and session['u_info'][-1]:
					sdate_str = session['u_info'][-1] 
				else:
					# Fallback if startdate missing
					sdate_str = track_date
					
				sdate = datetime.strptime(sdate_str, '%Y-%m-%d').date()
				edate = datetime.strptime(track_date, '%Y-%m-%d').date()
				delta = edate - sdate     

				for i in range(delta.days + 1):
					day = sdate + timedelta(days=i)
					days.append(str(day))

				split_list = [days[x:x+7] for x in range(0, len(days), 7)]
				# Safely determine week number
				try:
					this_weeknum = split_list.index(split_list[-1])+1
				except ValueError:
					this_weeknum = 1
					
				week_weights = []
				
				try:
					# Use fresh cursor or existing one
					for i in split_list[this_weeknum-1]:
						cur.execute("select * from progress where u_id=%s and p_date=%s ",(session['uid'],i,))
						day_data = cur.fetchall()
						if day_data:
							for row in day_data:
								week_weights.append(row[2])
		
					if len(week_weights) > 0:
						week_weight = round(sum(week_weights)/len(week_weights))
					else:
						week_weight = int(weight_kg)

					# Update or Insert progress_week
					cur.execute("update progress_week set pw_weight=%s where pw_num=%s and u_id=%s",(week_weight,this_weeknum,session['uid'],))
					if cur.rowcount == 0:
						cur.execute("insert into progress_week (u_id, pw_num, pw_weight) values (%s,%s,%s)", (session['uid'], this_weeknum, week_weight))
					
				except Exception as ex:
					print(f"Error calculating week stats in profile edit: {ex}", flush=True)

				conn.commit()
					
		except Exception as e:
			print(f"Error updating profile: {e}", flush=True)
			return (f'{e}')
		finally:
			if 'conn' in locals():
				conn.close()

		return redirect(url_for('profile'))

@app.route("/add_food", methods =['GET','POST'] )
def add_food():
	if request.method == 'GET':
		
		return render_template('add_food.html')

	else:
		mealtime = request.form['mealtime']
		
		
		return render_template('add_food.html',mealtime=mealtime)

def food_list(meal,item):
	
	meal.append(item)

	return meal

def one_food(item,portion,p_type):
	
	meal = [item,portion,p_type]

	return meal

@app.route("/add_successful", methods =['GET','POST'] )

def add_successful():
	
	if request.method == 'GET':
		
		return render_template('add_food.html')

	else:
		mealtime = request.form['mealtime']
		item_name = request.form['food']
		foodPortion = int(request.form['portion'])
		portion_type = request.form['portion_type']

		query = f"{foodPortion} {portion_type} {item_name}"
		info = call_spoonacular(query)

# Check if the request was successful
	if info:
		# Extract nutritional information from the response
		# Spoonacular already scales nutrition by amount in the response usually, 
		# but our helper returns raw per-serving data if "servings=1" is passed.
		# However, parseIngredients with "100g chicken" returns nutrtion FOR 100g.
		# So the info returned IS the final calculated info for that query string.
		
		name = info['food']
		calories = info['calories']
		protein = info['protein']
		carb = info['carbs']
		fat = info['fat']
		quantity = foodPortion
		unit = info['unit']
		if not unit:
			unit = portion_type

# Since Spoonacular parses "200g chicken" and returns nutrition for 200g, 
# we don't need to manually multiply by quantity if the query included it.
# But wait, looking at the code below, it calculates final values based on quantity.
# If we pass "1 cup chicken", Spoonacular gives nutrition for 1 cup.
# If logic below assumes `calories` is per unit?
# Logic below: finalCalorie = (calories * foodPortion) / quantity -> final = calories
# So if `calories` is already total, this math is redundant but harmless IF calories is total.
# BUT: The existing code logic implies `calories` from API might be per serving and they scale it?
# Nutritionix returns data for the specific query quantity usually.
# Let's trust that `call_spoonacular` returns the correct totals for the query.
		
		finalCalorie = calories
		finalProtein = protein
		finalCarb = carb
		finalFat = fat

# Store the information in a list or any preferred format
		food = [item_name, quantity, unit, finalCalorie, finalProtein, finalCarb, finalFat]
		uid = session['uid']
		track_date = str(datetime.today().strftime('%Y-%m-%d'))
	else:
		# Handle errors, if any
		print("Error: Failed to fetch data from Spoonacular")
		flash(f"Could not find food item. Please Check Spelling.")
		return render_template('add_food.html', mealtime=mealtime)
	try:
		with get_connection() as conn:
			cur = conn.cursor()
					
			cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid,))
			track_info= cur.fetchall()
			

			calorie = round(Decimal(track_info[0][6]), 2)+round(Decimal(food[3]), 2)
			protein = round(Decimal(track_info[0][7]), 2)+round(Decimal(food[4]), 2)
			carb = round(Decimal(track_info[0][8]), 2)+round(Decimal(food[5]), 2)
			fat = round(Decimal(track_info[0][9]), 2)+round(Decimal(food[6]), 2)
					
			cur2 = conn.cursor()
			
			if mealtime == "Breakfast":
				meal_input = track_info[0][2] + food[0] +","
				
				cur2.execute("update tracking set track_breakfast=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
				conn.commit()


			if mealtime == "Lunch":
				meal_input = track_info[0][3] + food[0] +","
				
				
				cur2.execute("update tracking set track_lunch=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
				conn.commit()

			if mealtime == "Snack":
				meal_input = track_info[0][4] + food[0] +","
				
				cur2.execute("update tracking set track_snack=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
				conn.commit()

			if mealtime == "Dinner":
				meal_input = track_info[0][5] + food[0] +","
				
				cur2.execute("update tracking set track_dinner=%s,track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
				conn.commit()
			

	except Exception as e:
		return (f'{e}')
	finally:
		conn.close()

	
	if mealtime == "Breakfast":
		
		session['bf_numbers'] = [0,0,0,0]
		session['bf_meal'] = food_list(bf_meal,food)
		
		for i in session['bf_meal']:
					
			session['bf_numbers'][0]+=round(Decimal(i[3]), 2)
			session['bf_numbers'][1]+=round(Decimal(i[4]), 2)
			session['bf_numbers'][2]+=round(Decimal(i[5]), 2)
			session['bf_numbers'][3]+=round(Decimal(i[6]), 2)
		print(session['bf_meal'])
				

	if mealtime == "Lunch":
		session['lunch_numbers'] = [0,0,0,0]
		session['lunch_meal'] = food_list(lunch_meal,food)

		for i in session['lunch_meal']:			
			session['lunch_numbers'][0]+=round(Decimal(i[3]), 2)
			session['lunch_numbers'][1]+=round(Decimal(i[4]), 2)
			session['lunch_numbers'][2]+=round(Decimal(i[5]), 2)
			session['lunch_numbers'][3]+=round(Decimal(i[6]), 2)
			
	
	if mealtime == "Snack":
		session['snack_numbers'] = [0,0,0,0]
		session['snack_meal'] = food_list(snack_meal,food)

		for i in session['snack_meal']:
					
			session['snack_numbers'][0]+=round(Decimal(i[3]), 2)
			session['snack_numbers'][1]+=round(Decimal(i[4]), 2)
			session['snack_numbers'][2]+=round(Decimal(i[5]), 2)
			session['snack_numbers'][3]+=round(Decimal(i[6]), 2)

	if mealtime == "Dinner":
		session['dinner_numbers'] = [0,0,0,0]
		session['dinner_meal'] = food_list(dinner_meal,food)

		for i in session['dinner_meal']:
					
			session['dinner_numbers'][0]+=round(Decimal(i[3]), 2)
			session['dinner_numbers'][1]+=round(Decimal(i[4]), 2)
			session['dinner_numbers'][2]+=round(Decimal(i[5]), 2)
			session['dinner_numbers'][3]+=round(Decimal(i[6]), 2)
	

	return render_template('add_food.html',mealtime=mealtime)

@app.route("/track", methods = ['GET','POST'])
def track():
	if request.method == 'GET':
		uid = session['uid']
		
		track_date = str(datetime.today().strftime ('%Y-%m-%d'))

		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid))
				track_info= cur.fetchall()


				if track_info:
					pass
				else:
					try:
						with get_connection() as conn:
							cur = conn.cursor()
							breakfast=""
							lunch = ""
							dinner = ""
							snack = ""
							cur.execute("insert into tracking (track_date,u_id,track_calorie,track_protein,track_fat,track_carb,track_breakfast,track_lunch,track_snack,track_dinner) values (%s,%s,0,0,0,0,%s,%s,%s,%s)",(track_date,uid,breakfast,lunch,snack,dinner))
							conn.commit()

							cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid))
							track_info= cur.fetchall()
							if track_info[0][2] == ",":
								cur.execute("update tracking set track_breakfast=%s",("",))
								conn.commit()
					except Exception as e:
						return (f'{e}')
					finally:
						conn.close()

				cur.execute("select * from tracking where track_date=%s and u_id=%s",(track_date,uid))
				track_info= cur.fetchall()

				protein_goal = session['u_info'][14]
				carb_goal = session['u_info'][15]
				fat_goal = session['u_info'][16]
				breakfast = track_info[0][2]
				lunch = track_info[0][3]
				snack = track_info[0][4]
				dinner = track_info[0][5]

				# Parse into lists for display
				breakfast_list = [x.strip() for x in breakfast.split(',') if x.strip()]
				lunch_list = [x.strip() for x in lunch.split(',') if x.strip()]
				snack_list = [x.strip() for x in snack.split(',') if x.strip()]
				dinner_list = [x.strip() for x in dinner.split(',') if x.strip()]

					
				protein_consumed = track_info[0][7]
				carb_consumed = track_info[0][8]
				fat_consumed = track_info[0][9]

				calorie_goal = session['u_info'][17]
				calorie_consumed = track_info[0][6]
				
				# Auto-repair negative values
				if protein_consumed < 0: protein_consumed = 0
				if carb_consumed < 0: carb_consumed = 0
				if fat_consumed < 0: fat_consumed = 0
				if calorie_consumed < 0: calorie_consumed = 0

				# Auto-repair phantom values (if lists empty, but DB has values)
				if not breakfast_list and not lunch_list and not snack_list and not dinner_list:
					protein_consumed = 0.0
					carb_consumed = 0.0
					fat_consumed = 0.0
					calorie_consumed = 0.0

				protein_percent = "{:.2f}".format((protein_consumed/protein_goal) * 100)
				carb_percent = "{:.2f}".format((carb_consumed/carb_goal) * 100)
				fat_percent = "{:.2f}".format((fat_consumed/fat_goal) * 100)
				calorie_percent = "{:.2f}".format((calorie_consumed/calorie_goal) * 100)
					
				try:
					with get_connection() as conn:
						cur = conn.cursor()
						cur.execute("update tracking set track_calorie=%s,track_protein=%s,track_carb=%s,track_fat=%s where track_date=%s and u_id=%s", (calorie_consumed,protein_consumed,carb_consumed,fat_consumed,track_date,uid,))
						conn.commit()

				except Exception as e:
					return (f'{e}')
				finally:
					conn.close()
					
				return render_template('track.html',p_goal=protein_goal,
											c_goal = carb_goal,
											f_goal=fat_goal,
											p_consumed = protein_consumed,
											c_consumed=carb_consumed,
											f_consumed = fat_consumed,
											p_percent = protein_percent,
											c_percent = carb_percent,
											f_percent = fat_percent,
											cal_percent = calorie_percent,
											cal_goal = calorie_goal,
											cal_consumed = calorie_consumed,
											breakfast = breakfast,
											lunch = lunch,
											snack = snack,
											dinner = dinner,

											breakfast_list=breakfast_list,
											lunch_list=lunch_list,
											snack_list=snack_list,
											dinner_list=dinner_list,
											)
						
		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		
	else:
		return render_template('track.html')

@app.route("/delete_track_item", methods=['POST'])
def delete_track_item():
	if 'uid' not in session:
		return redirect(url_for('login'))
		
	uid = session['uid']
	mealtime = request.form.get('mealtime')
	item_name = request.form.get('item_name', '').strip()
	today = datetime.today().strftime('%Y-%m-%d')
	
	if not mealtime or not item_name:
		return redirect(url_for('track'))

	# Session keys mapping
	session_map = {
		"Breakfast": ("bf_meal", "bf_numbers"),
		"Lunch": ("lunch_meal", "lunch_numbers"),
		"Snack": ("snack_meal", "snack_numbers"),
		"Dinner": ("dinner_meal", "dinner_numbers")
	}

	try:
		with get_connection() as conn:
			cur = conn.cursor()
			cur.execute("SELECT * FROM tracking WHERE track_date=%s AND u_id=%s", (today, uid))
			row = cur.fetchone()

			if not row:
				cur.execute("SELECT * FROM tracking WHERE track_date=%s AND u_id=%s", (today, uid))
				row = cur.fetchone()

			if row:
				# Initialize subtract values
				cal, prot, carb, fat = 0, 0, 0, 0
				found_in_session = False

				# 1. Try to find/remove from Session first (Preferred for accuracy & UI sync)
				if mealtime in session_map:
					list_key, numbers_key = session_map[mealtime]
					
					# session[list_key] is a list of lists: [name, qty, unit, cal, prot, carb, fat]
					if list_key in session and session[list_key]:
						# Find index of item
						idx_to_remove = -1
						for i, food_item in enumerate(session[list_key]):
							# food_item[0] is name
							if food_item[0] == item_name:
								idx_to_remove = i
								# Get exact values used
								cal = float(food_item[3])
								prot = float(food_item[4])
								carb = float(food_item[5])
								fat = float(food_item[6])
								found_in_session = True
								break
						
						if idx_to_remove != -1:
							# Remove from session list
							session[list_key].pop(idx_to_remove)
							# Update session numbers [cal, prot, carb, fat]
							if numbers_key in session and session[numbers_key]:
								session[numbers_key][0] = max(0, session[numbers_key][0] - cal)
								session[numbers_key][1] = max(0, session[numbers_key][1] - prot)
								session[numbers_key][2] = max(0, session[numbers_key][2] - carb)
								session[numbers_key][3] = max(0, session[numbers_key][3] - fat)
							
							session.modified = True

				# 2. If not in session, fallback to API/Estimate
				if not found_in_session:
					info = call_spoonacular(item_name)
					if info:
						cal = info.get('calories', 0)
						prot = info.get('protein', 0)
						carb = info.get('carbs', 0)
						fat = info.get('fat', 0)

				# 3. Update Database Totals
				curr_cal = float(row['track_calorie']) if row['track_calorie'] else 0
				curr_prot = float(row['track_protein']) if row['track_protein'] else 0
				curr_carb = float(row['track_carb']) if row['track_carb'] else 0
				curr_fat = float(row['track_fat']) if row['track_fat'] else 0

				new_cal = max(0, curr_cal - cal)
				new_prot = max(0, curr_prot - prot)
				new_carb = max(0, curr_carb - carb)
				new_fat = max(0, curr_fat - fat)

				# 4. Update Database Meal String
				col_map = {
					"Breakfast": "track_breakfast",
					"Lunch": "track_lunch",
					"Snack": "track_snack",
					"Dinner": "track_dinner"
				}
				
				db_col = col_map.get(mealtime)
				if db_col:
					current_items_str = row[db_col] if row[db_col] else ""
					items = [x.strip() for x in current_items_str.split(',') if x.strip()]
					
					if item_name in items:
						items.remove(item_name)
						
						new_items_str = ",".join(items)
						if new_items_str:
							new_items_str += "," 
							
						cur.execute(f"UPDATE tracking SET {db_col}=%s, track_calorie=%s, track_protein=%s, track_carb=%s, track_fat=%s WHERE track_date=%s AND u_id=%s",
									(new_items_str, new_cal, new_prot, new_carb, new_fat, today, uid))
						conn.commit()

	except Exception as e:
		print(f"Error deleting item: {e}")
		pass

	return redirect(url_for('track'))


@app.route("/register", methods = ['GET','POST'])
def citizen_register():
	if request.method == 'GET':
		return render_template('register.html')
	else:
		name = request.form['name']
		
		email = request.form['email']
		password = request.form['password']
		
		return render_template('profilesetup.html',name=name,email=email,password=password)

@app.route("/login", methods = ['GET','POST'])
def login():
	if request.method == 'GET':
		return render_template('login.html')
	else:
		
		session['uid'] = 0
		email = request.form['email']
		password = request.form['password']
		secure_pwd = pwd_encode(password)
		msg=''
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_email=%s",(email,))
				u_info= cur.fetchall()
				if not u_info:
					flash(f'The email address ({email}) that you entered does not exist in our database.')
					return redirect(url_for('login'))
				else:
					for row in u_info:
						session['uid'] = row[0]
						u_pass = row[2] 
						u_name = row[1]
						u_date = row[-1]
					
					if secure_pwd == u_pass:
						days = []
						flash(f'Your have successfully logged in as {u_name}')
						session['u_logged'] = True
						session['u_info'] = []
						session['u_pass'] = password 

						track_date = datetime.today().strftime ('%Y-%m-%d')
						sdate = datetime.strptime(u_date, '%Y-%m-%d').date()
						edate = datetime.strptime(track_date, '%Y-%m-%d').date()
						delta = edate - sdate     

						for i in range(delta.days + 1):
							day = sdate + timedelta(days=i)
							days.append(str(day))
							journey = len(days)

						try:
							with get_connection() as conn:
								cur = conn.cursor()
								cur2 = conn.cursor()
								cur2.execute("update user set u_journey=%s where u_id=%s", (journey,session['uid'],))
								conn.commit()

								cur.execute("select * from user where u_id=%s",(session['uid'],))
								u_info = cur.fetchone()
				
								for row in u_info:
									session['u_info'].append(row)

						except Exception as e:
							return (f'{e}')
						finally:
							conn.close()

						return redirect(url_for('index'))
					else:
						session.pop('uid',None)
						flash('Sorry the credentails you are using are invalid')
						return redirect(url_for('login'))

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()

@app.route("/setup", methods = ['GET','POST'])
def profilesetup():
	if request.method == 'GET':
		return render_template('profilesetup.html')

	else:
		name = request.form['name']
		email = request.form['email']
		passwd = request.form['password']
		password = pwd_encode(passwd)
		age = int(request.form['age'])
		gender = request.form['gender']
		vegan = request.form['vegan']
		allergy = request.form['allergy']
		weight_kg = int(request.form['weight'])
		feet = int(request.form['feet'])
		inches = int(request.form['inches'])
		activity_level = request.form['activity']
		height_bmi = int((feet * 12) + inches)
		height_m = height_bmi * 0.0254
		bmr = 0
		body_status = ""
		# Metric BMI: kg/m^2
		BMI =  weight_kg / (height_m * height_m)
		bodyfat = 0

		if gender == "male":
			bmr = int((10 * weight_kg) + (15.88 * height_bmi) - (5 * age) + 5)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 16.2)
		else:
			bmr = int((10 * weight_kg) + (15.88 * height_bmi) - (5 * age) - 161)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 5.4)

		calorie = 0

		if activity_level == "sedentary":
			calorie = int(bmr*1.2)

		elif activity_level == "lightly active":
			calorie = int(bmr * 1.375)

		elif activity_level == "moderately active":
			calorie = int(bmr * 1.55)

		elif activity_level == "very active":
			calorie = int(bmr * 1.725)

		elif activity_level == "extra active":
			calorie = int(bmr * 1.9)

		if BMI < 18.5:
			body_status = "underweight"

		elif BMI >= 18.5 and BMI <= 24.9 :
			body_status = "healthy weight"

		elif BMI >= 25 and BMI <= 29.9 :
			body_status = "overweight"

		elif BMI >= 30 :
			body_status = "obese"

		protein = int(((calorie-500) * 0.30)/4)
		carb = int(((calorie-500)* 0.40)/4)
		fat = int(((calorie-500) * 0.30)/9)
		fiber = int(calorie/1000*14)
		journey = 1
		breakfast = int((calorie-500) * 0.30)
		snack = int((calorie-500)* 0.10)
		lunch = int((calorie-500)* 0.35)
		dinner = int((calorie-500)* 0.25)

		try:
			with get_connection() as conn:
				db = conn.cursor()
				db.execute("insert into user (u_username,u_email,u_password,u_age,u_gender,u_vegan,u_allergy,u_weight,u_feet,u_inches,u_bmi,u_activitylevel,u_protein,u_carb,u_fat,u_fiber,u_calories,u_journey,u_bodyfat,u_status,u_startdate) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(name,email,password,age,gender,vegan,allergy,weight_kg,feet,inches,int(BMI),activity_level,protein,carb,fat,fiber,calorie,journey,bodyfat,body_status,datetime.today().strftime ('%Y-%m-%d'),))
				conn.commit()
				flash('Successfully Registered')

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()

	
		return redirect(url_for('login'))

@app.route("/profile", methods = ['GET','POST'])
def profile():
	if request.method == 'GET':
		uid = session['uid']
		try:
			with get_connection() as conn:
				db = conn.cursor()
				db.execute("select * from user where u_id=%s",(uid,))
				u_info = db.fetchone()

				

				
		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()

		return render_template('profile.html',u_info = u_info)

	else:
		return render_template('profile.html')

@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():

    import pickle
    import pandas as pd
    import numpy as np
    from flask import session

    # ================= INIT SESSION =================
    if "meal_plan" not in session:
        session["meal_plan"] = {
            "Breakfast": [],
            "Lunch": [],
            "Snack": [],
            "Dinner": []
        }

    if request.method == "GET":
        return render_template(
            "recommendation.html",
            meal_plan=session["meal_plan"]
        )

    # ================= LOAD FILES =================
    # Use relative paths for deployment compatibility
    model = pickle.load(open(str(BASE_DIR / "models" / "meal_xgb_model.pkl"), "rb"))
    encoder = pickle.load(open(str(BASE_DIR / "models" / "meal_label_encoder.pkl"), "rb"))
    feature_cols = pickle.load(open(str(BASE_DIR / "models" / "x_feature_columns.pkl"), "rb"))

    df = pd.read_csv(str(BASE_DIR / "dietdataset_new.csv"))
    df["meal_name"] = df["meal_name"].str.lower().str.strip()

    # ================= USER INPUT =================
    meal_time = request.form.get("meal_time")   # Breakfast/Lunch/Snack/Dinner
    carb_base = request.form.get("carb_base")
    protein_source = request.form.get("protein_source")
    primary_veggie = request.form.get("primary_veggie")
    veg_type = request.form.get("veg_type")     # veg / non-veg
    allergy = request.form.get("allergy", "none").lower()

    # ================= FEATURE VECTOR =================
    row = {
        "meal_time": meal_time,
        "carb_base": carb_base,
        "protein_source": protein_source,
        "primary_veggie": primary_veggie,
        "veg_type": veg_type,
        "allergy": allergy
    }

    X = pd.DataFrame([row])
    X = pd.get_dummies(X)
    X = X.reindex(columns=feature_cols, fill_value=0)

    # ================= TOP-2 PREDICTION =================
    probs = model.predict_proba(X)[0]
    top2_idx = np.argsort(probs)[-2:][::-1]
    meals = encoder.inverse_transform(top2_idx)

    # ================= FILTER + NUTRITION =================
    results = []

    for meal in meals:
        meal_l = meal.lower().strip()
        rows = df[df["meal_name"] == meal_l]

        if rows.empty:
            continue

        r = rows.iloc[0]

        # ---- HARD FILTERS ----
        if r["meal_time"] != meal_time:
            continue
        if veg_type == "veg" and r["veg_type"] != "veg":
            continue
        if allergy != "none" and r["allergy"] == allergy:
            continue

        # ---- Nutrition via API ----
        api_data = call_spoonacular(meal)

        if api_data and api_data.get("calories", 0) > 0:
            nut = {
                "cal": int(api_data["calories"]),
                "protein": int(api_data["protein"]),
                "carb": int(api_data["carbs"]),
                "fat": int(api_data["fat"])
            }
        else:
            # ---- Dataset fallback ----
            nut = {
                "cal": int(r["calories"]),
                "protein": int(r["protein_g"]),
                "carb": int(r["carb_g"]),
                "fat": int(r["fat_g"])
            }

        results.append({
            "name": meal.title(),
            "nutrition": nut
        })

    # ================= SAVE TO SESSION (CRITICAL) =================
    session["meal_plan"][meal_time] = results
    session.modified = True

    return render_template(
        "recommendation.html",
        meal_plan=session["meal_plan"]
    )



@app.route("/clear_recommendation", methods=["POST"])
def clear_recommendation():
    session.pop("meal_plan", None)
    return redirect(url_for("recommendation"))


@app.route("/recommend_setup", methods = ['GET','POST'])
def recommend_setup():
    if request.method == 'GET':
        print(session['u_info'][12])
        return render_template('recommendsetup.html')
    else:
        return render_template('recommendsetup.html')

@app.route("/progress", methods = ['GET','POST'])
def progress():
	if request.method == 'GET':
		uid = session['uid']
		track_date = datetime.today().strftime ('%Y-%m-%d')
		days = []
		display_day = []
		weeks = []
		day_weight = []
		week_weight = []
		sdate = datetime.strptime(session['u_info'][-1], '%Y-%m-%d').date()
		edate = datetime.strptime(track_date, '%Y-%m-%d').date()
		delta = edate - sdate     

		for i in range(delta.days + 1):
			day = sdate + timedelta(days=i)
			days.append(str(day))

		
		
		split_list = [days[x:x+7] for x in range(0, len(days), 7)]
		weeknum = split_list.index(split_list[-1])+1
		pw_date = split_list[-1][-1]
		
		pw_weight =[]
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from progress where u_id=%s and p_date=%s",(uid,track_date))
				data = cur.fetchone()
				if not data:
					cur.execute("insert into progress (u_id,p_date,p_weight) values (%s,%s,%s)",(session['u_info'][0],track_date,session['u_info'][6]))
					conn.commit()

				cur.execute("select * from progress where u_id=%s and p_date=%s",(uid,track_date))
				data2 = cur.fetchone()	
				for i in data2:
					pw_weight.append(data[2])
					

				cur.execute("select * from progress_week where u_id=%s and pw_num=%s",(uid,weeknum))
				week_exist = cur.fetchone()

				if not week_exist:
					
						
					cur.execute("insert into progress_week (u_id,pw_num,pw_weight) values (%s,%s,%s)",(session['u_info'][0],weeknum,pw_weight[0]))
					conn.commit()



				
					
			cur.execute("select * from progress_week where u_id=%s",(uid,))
			week_data = cur.fetchall()
			for i in week_data:
				weeks.append("Week"+str(i[1]))
				week_weight.append(i[2])
			for i in days:
				cur.execute("select * from progress where u_id=%s and p_date=%s",(uid,i))
				get_weight = cur.fetchall()
				for i in get_weight:
					day_weight.append(i[2])

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from progress where u_id=%s",(uid,))
				dates = cur.fetchall()
				for i in dates:
					getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
					dates = getdate.strftime("%B-%d")
			
					display_day.append(dates)

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()

		
			

		return render_template('progress.html',days = display_day[-7:],weeks = weeks[-7:],d_weight = day_weight[-7:],w_weight = week_weight[-7:])
		

	else:
		return render_template('progress.html')

@app.route("/daily_detail", methods = ['GET','POST'])
def daily_detail():
	if request.method == 'GET':
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from tracking where u_id=%s and track_date=%s",(session['uid'],datetime.today().strftime ('%Y-%m-%d')))
				i = cur.fetchone()

			# Check if data exists for today
			if not i:
				# Return template with error message displayed inline
				return render_template('daily_detail.html', 
					date = datetime.today().strftime("%B-%d-%Y"), 
					breakfast = "", lunch = "", snack = "", dinner = "", 
					calorie = session['u_info'][17], protein = 0, carb = 0, fat = 0, 
					consumed = 0, result = "", deficit = "Calorie Deficit: 0kcal", 
					weight = session['u_info'][6],
					error_message = "No tracking data found for today. Please track some meals first.")


			getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
			date = getdate.strftime("%B-%d-%Y")
			breakfast = i[2]
			lunch = i[3]
			snack = i[4]
			dinner = i[5]
			calorie = session['u_info'][17]
			protein = i[7]
			carb = i[8]
			fat = i[9]
			consumed = i[6]
			deficit = round(Decimal(calorie - i[6]), 2)
			result = "Reduced "+ str(round(Decimal(consumed/7700),4))+"kg of bodyweight (in theory)"
			deficits = "Calorie Deficit: "+ str(deficit) +"kcal"

			cur.execute("select * from progress where u_id=%s and p_date=%s",(session['uid'],datetime.today().strftime ('%Y-%m-%d'),))
			weights = cur.fetchone()
			if weights:
				for i in weights:
					weight = weights[2]
			else:
				weight = session['u_info'][6]

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('daily_detail.html',date = date,
												   breakfast = breakfast,
												   lunch = lunch,
												   snack = snack,
												   dinner = dinner,
												   calorie = calorie,
												   protein = protein,
												   carb = carb,
												   fat = fat,
												   consumed = consumed,
												   result = result,
												   deficit = deficits,
												   weight = weight)

	else:
		getdate = request.form['date']
		weight = ""
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from tracking where u_id=%s and track_date=%s",(session['uid'],getdate,))
				i = cur.fetchone()
				
				# Check if data exists for the selected date
				if not i:
					flash(f'No tracking data found for the selected date ({getdate}). Please select a date where you have tracked meals.', 'warning')
					return render_template('daily_detail.html',
					date = datetime.strptime(getdate, '%Y-%m-%d').strftime("%B-%d-%Y"),
					breakfast = "", lunch = "", snack = "", dinner = "",
					calorie = session['u_info'][17], protein = 0, carb = 0, fat = 0,
					consumed = 0, result = "", deficit = "Calorie Deficit: 0kcal",
					weight = session['u_info'][6],
					error_message = f"No tracking data found for the selected date ({getdate}). Please select a date where you have tracked meals.")
				
				print(i[1])
				


				
				getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
				date = getdate.strftime("%B-%d-%Y")
				breakfast = i[2]
				lunch = i[3]
				snack = i[4]
				dinner = i[5]
				calorie = int(session['u_info'][17])
				protein = i[7]
				carb = i[8]
				fat = i[9]
				consumed = i[6]
				deficit = round(Decimal(calorie - i[6]), 2)
				result = "Reduced "+ str(round(Decimal(consumed/7700),4))+"kg of bodyweight (in theory)"
				deficits = "Calorie Deficit: "+ str(deficit) +"kcal"

				cur.execute("select * from progress where u_id=%s and p_date=%s",(session['uid'],getdate,))
				weights = cur.fetchone()
				if weights:
					for i in weights:
						weight = weights[2]
				else:
					weight = session['u_info'][6]
		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('daily_detail.html',date = date,
												   breakfast = breakfast,
												   lunch = lunch,
												   snack = snack,
												   dinner = dinner,
												   calorie = calorie,
												   protein = protein,
												   carb = carb,
												   fat = fat,
												   consumed = consumed,
												   result = result,
												   deficit = deficits,
												   weight = weight)

@app.route("/weekly_detail", methods = ['GET','POST'])
def weekly_detail():
	if request.method == 'GET':
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from progress_week where u_id=%s",(session['uid'],))
				u_week = cur.fetchall()
				weeks = []
				days = []
				track_date = datetime.today().strftime ('%Y-%m-%d')
				for i in u_week:
					weeks.append(i[1])

				sdate = datetime.strptime(session['u_info'][-1], '%Y-%m-%d').date()
				edate = datetime.strptime(track_date, '%Y-%m-%d').date()
				delta = edate - sdate     

				for i in range(delta.days + 1):
					day = sdate + timedelta(days=i)
					days.append(str(day))

		
				split_list = [days[x:x+7] for x in range(0, len(days), 7)]
				this_weeknum = split_list.index(split_list[-1])+1
				print(split_list[this_weeknum-1])
				
				calories = []
				proteins = []
				fats = []
				carbs = []
				
				try:
					with get_connection() as conn:
						cur = conn.cursor()
						cur.execute("select * from progress_week where u_id=%s and pw_num=%s",(session['uid'],this_weeknum,))
						wow = cur.fetchone()
						weight_of_week = wow[2]
						for i in split_list[this_weeknum-1]:
							cur.execute("select * from tracking where u_id=%s and track_date=%s",(session['uid'],i,))
							u_week = cur.fetchall()
							for i in u_week:
							
								calories.append(i[6])
								proteins.append(i[7])
								carbs.append(i[8])
								fats.append(i[9])


						calorie_consumed = sum(calories)
						
						required = float(session['u_info'][17])*len(calories)
						
						calorie_required = required

						deficit = calorie_required - calorie_consumed

						# Check if we have any tracking data to avoid division by zero
						if len(calories) == 0:
							# Return template with error message displayed inline
							return render_template('weekly_detail.html', weeks = weeks, average_calorie = 0, average_carb = 0, average_fat = 0, average_protein = 0, net_deficit = 0, result = "No data available", average_deficit = 0, week = this_weeknum, weight_week = weight_of_week, error_message = "No tracking data available for this week. Please track some meals first.")

						average_calorie = round(Decimal(sum(calories)/len(calories)),2)
						average_protein = round(Decimal(sum(proteins)/len(proteins)),2)
						average_carb = round(Decimal(sum(carbs)/len(carbs)),2)
						average_fat = round(Decimal(sum(fats)/len(fats)),2)
						average_deficit = round(Decimal(deficit/len(calories)),2)
						net_deficit = round(Decimal(deficit),2)
						loss_weight = round(Decimal(net_deficit/7700),2)
						result = "Reduced "+ str(loss_weight) +"kg of bodyweight in this whole week (in theory)"

				except Exception as e:
					return (f'{e}')
				finally:
					conn.close()
				
				

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('weekly_detail.html',weeks = weeks,
													average_calorie = average_calorie,
													average_carb = average_carb,
													average_fat = average_fat,
													average_protein = average_protein,
													net_deficit = net_deficit,
													result = result,
													average_deficit = average_deficit,
													week = this_weeknum,
													weight_week = weight_of_week
												    )

	else:
		getweek = request.form['weeks']
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from progress_week where u_id=%s and pw_num=%s",(session['uid'],getweek))
				u_week = cur.fetchall()
				cur.execute("select * from progress_week where u_id=%s",(session['uid'],))
				u_weeks = cur.fetchall()
				weeks = []
				days = []
				
				for i in u_weeks:
					weeks.append(i[1])

				track_date = datetime.today().strftime ('%Y-%m-%d')
				sdate = datetime.strptime(session['u_info'][-1], '%Y-%m-%d').date()
				edate = datetime.strptime(track_date, '%Y-%m-%d').date()
				delta = edate - sdate     

				for i in range(delta.days + 1):
					day = sdate + timedelta(days=i)
					days.append(str(day))

		
				split_list = [days[x:x+7] for x in range(0, len(days), 7)]
				this_weeknum = int(getweek)
				
				
				calories = []
				proteins = []
				fats = []
				carbs = []
				
				try:
					with get_connection() as conn:
						cur = conn.cursor()
						cur.execute("select * from progress_week where u_id=%s and pw_num=%s",(session['uid'],this_weeknum,))
						wow = cur.fetchone()
						weight_of_week = wow[2]
						for i in split_list[this_weeknum-1]:
							cur.execute("select * from tracking where u_id=%s and track_date=%s and track_calorie!=%s",(session['uid'],i,0))
							u_week = cur.fetchall()
							for i in u_week:
							
								calories.append(i[6])
								proteins.append(i[7])
								carbs.append(i[8])
								fats.append(i[9])


						calorie_consumed = sum(calories)
						
						required = float(session['u_info'][17])*len(calories)
						
						calorie_required = required

						deficit = calorie_required - calorie_consumed

						# Check if we have any tracking data to avoid division by zero
						if len(calories) == 0:
							# Return template with error message displayed inline
							return render_template('weekly_detail.html', weeks = weeks, average_calorie = 0, average_carb = 0, average_fat = 0, average_protein = 0, net_deficit = 0, result = "No data available", average_deficit = 0, week = getweek, weight_week = weight_of_week, error_message = "No tracking data available for the selected week. Please track some meals first.")

						average_calorie = round(Decimal(sum(calories)/len(calories)),2)
						average_protein = round(Decimal(sum(proteins)/len(proteins)),2)
						average_carb = round(Decimal(sum(carbs)/len(carbs)),2)
						average_fat = round(Decimal(sum(fats)/len(fats)),2)
						average_deficit = round(Decimal(deficit/len(calories)),2)
						net_deficit = round(Decimal(deficit),2)
						loss_weight = round(Decimal(net_deficit/7700),2)
						result = "Reduced "+ str(loss_weight) +"kg of bodyweight in this whole week (in theory)"

				except Exception as e:
					return (f'{e}')
				finally:
					conn.close()
				
				

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('weekly_detail.html',weeks = weeks,
													average_calorie = average_calorie,
													average_carb = average_carb,
													average_fat = average_fat,
													average_protein = average_protein,
													net_deficit = net_deficit,
													result = result,
													average_deficit = average_deficit,
													week = getweek,
													weight_week = weight_of_week,
												    )


@app.route("/index", methods = ['GET','POST'])
def index():
	if request.method == 'GET':

		try:	
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_id=%s",(session['uid'],))
				u_data = cur.fetchone()
				weight = u_data[6]


		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('index.html', weight = weight)

	else:
		getweight = request.form['weight']
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("update user set u_weight=%s where u_id=%s",(getweight,session['uid'],))
				conn.commit()

				cur.execute("update progress set p_weight=%s where p_date=%s and u_id=%s",(getweight,datetime.today().strftime ('%Y-%m-%d'),session['uid'],))
				if cur.rowcount == 0:
					cur.execute("insert into progress (u_id,p_date,p_weight) values (%s,%s,%s)", (session['uid'], datetime.today().strftime('%Y-%m-%d'), getweight))
				conn.commit()

				cur.execute("select * from user where u_id=%s",(session['uid'],))
				u_data = cur.fetchone()
				weight = u_data[6]

				days = []
				weeks = []
				track_date = datetime.today().strftime ('%Y-%m-%d')
				sdate = datetime.strptime(session['u_info'][-1], '%Y-%m-%d').date()
				edate = datetime.strptime(track_date, '%Y-%m-%d').date()
				delta = edate - sdate     

				for i in range(delta.days + 1):
					day = sdate + timedelta(days=i)
					days.append(str(day))

				split_list = [days[x:x+7] for x in range(0, len(days), 7)]
				this_weeknum = split_list.index(split_list[-1])+1
				week_weights = []
				print(this_weeknum)
				try:
					with get_connection() as conn:
						cur = conn.cursor()
						for i in split_list[this_weeknum-1]:
							cur.execute("select * from progress where u_id=%s and p_date=%s ",(session['uid'],i,))
							day_data = cur.fetchall()
							if day_data:
								for row in day_data:
									# Assuming row[2] is weight based on previous code usage
									week_weights.append(row[2])
			
					if len(week_weights) > 0:
						week_weight = round(sum(week_weights)/len(week_weights))
					else:
						# Fallback to current weight if no weights found
						week_weight = int(getweight)

					# Update or Insert progress_week
					cur.execute("update progress_week set pw_weight=%s where pw_num=%s and u_id=%s",(week_weight,this_weeknum,session['uid'],))
					if cur.rowcount == 0:
						cur.execute("insert into progress_week (u_id, pw_num, pw_weight) values (%s,%s,%s)", (session['uid'], this_weeknum, week_weight))
					
					conn.commit()

				except Exception as e:
					print(f"Error in week calculation: {e}", flush=True)
				finally:
					conn.close()

				

		except Exception as e:
			return (f'{e}')
		finally:
			conn.close()
		return redirect(url_for('recommend_setup'))

@app.route("/about", methods =['GET'] )
def about():
	if request.method == 'GET':

		return render_template('about.html')

@app.route("/documentation")
def documentation():
	return render_template('documentation.html')

@app.route('/logout')
def logout():
	
	session.pop('uid',None)
	session.pop('u_pass',None)
	session.pop('u_info',None)
	session.pop('bf_meal',None)
	session.pop('bf_numbers',None)
	session.pop('lunch_meal',None)
	session.pop('lunch_numbers',None)
	session.pop('dinner_meal',None)
	session.pop('dinner_numbers',None)
	session.pop('snack_meal',None)
	session.pop('snack_numbers',None)
	bf_meal.clear()
	lunch_meal.clear()
	snack_meal.clear()
	dinner_meal.clear()
	flash('You have successfully logged out')
	return redirect(url_for('login'))

if __name__ == "__main__":
	# Production-ready configuration
	port = int(os.environ.get('PORT', 8080))
	debug_mode = os.environ.get('FLASK_ENV', 'production') != 'production'
	
	print(f"Starting SmartDiet AI on port {port}...")
	print(f"Debug mode: {debug_mode}")
	print(f"Database: {'PostgreSQL' if os.environ.get('DATABASE_URL') else 'SQLite'}")
	
	app.run(host='0.0.0.0', port=port, debug=debug_mode)