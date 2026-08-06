import re
import random
import uuid

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Download NLTK resources (runs once) ─────────────────────────────────────
for _res in ["punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4"]:
    nltk.download(_res, quiet=True)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="NutriBot API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# NLTK PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

lemmatizer = WordNetLemmatizer()

# Keep negation words; they matter for intent (e.g. "not losing weight")
_STOP = set(stopwords.words("english")) - {
    "not", "no", "how", "much", "many", "need", "should", "do", "i", "my"
}


def preprocess(text: str) -> str:
    """Lowercase → tokenise → lemmatise → remove stop-words → rejoin."""
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in _STOP
    ]
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA  (sentences, labels in matching order)
# ═══════════════════════════════════════════════════════════════════════════════

_RAW: list[tuple[str, str]] = [
    # ── greeting ──────────────────────────────────────────────────────────────
    ("hello", "greeting"),
    ("hi there", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("what's up nutribot", "greeting"),
    # ── goodbye ───────────────────────────────────────────────────────────────
    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("see you later", "goodbye"),
    ("thanks bye", "goodbye"),
    ("that is all for now", "goodbye"),
    # ── diet_plan ─────────────────────────────────────────────────────────────
    ("give me a diet plan", "diet_plan"),
    ("what should I eat today", "diet_plan"),
    ("create my meal plan", "diet_plan"),
    ("veg diet plan for me", "diet_plan"),
    ("non veg meal plan", "diet_plan"),
    ("plan my meals for the day", "diet_plan"),
    ("suggest a healthy diet", "diet_plan"),
    ("I need a nutrition plan", "diet_plan"),
    ("design a diet for me", "diet_plan"),
    ("give me a food plan", "diet_plan"),
    # ── fat_loss ──────────────────────────────────────────────────────────────
    ("diet for fat loss", "fat_loss"),
    ("I need a fat loss diet", "fat_loss"),
    ("how to reduce weight", "fat_loss"),
    ("I want to lose weight", "fat_loss"),
    ("weight loss meal plan", "fat_loss"),
    ("help me slim down", "fat_loss"),
    ("I want to burn fat", "fat_loss"),
    ("reduce my body fat percentage", "fat_loss"),
    ("cutting diet plan", "fat_loss"),
    # ── muscle_gain ───────────────────────────────────────────────────────────
    ("diet for muscle gain", "muscle_gain"),
    ("I want to gain muscle", "muscle_gain"),
    ("bulking diet plan", "muscle_gain"),
    ("increase my muscle mass", "muscle_gain"),
    ("mass gain nutrition plan", "muscle_gain"),
    ("help me get bigger and stronger", "muscle_gain"),
    ("build muscle meal plan", "muscle_gain"),
    ("hypertrophy diet", "muscle_gain"),
    ("I want to bulk up", "muscle_gain"),
    # ── maintenance ───────────────────────────────────────────────────────────
    ("I want to maintain my weight", "maintenance"),
    ("maintenance calories please", "maintenance"),
    ("stay at current body weight", "maintenance"),
    ("keep my weight stable", "maintenance"),
    ("weight maintenance plan", "maintenance"),
    # ── calories ──────────────────────────────────────────────────────────────
    ("how many calories do I need", "calories"),
    ("calorie intake for fat loss", "calories"),
    ("calories for bulking", "calories"),
    ("what is my daily calorie requirement", "calories"),
    ("calculate my TDEE", "calories"),
    ("what is my calorie target", "calories"),
    ("calorie deficit for weight loss", "calories"),
    ("calorie surplus for muscle gain", "calories"),
    # ── protein ───────────────────────────────────────────────────────────────
    ("how much protein do I need", "protein"),
    ("what is my protein requirement", "protein"),
    ("daily protein intake recommendation", "protein"),
    ("protein intake for muscle building", "protein"),
    ("how many grams of protein per day", "protein"),
    # ── bmi ───────────────────────────────────────────────────────────────────
    ("what is my BMI", "bmi"),
    ("check my bmi please", "bmi"),
    ("am I overweight", "bmi"),
    ("calculate my body mass index", "bmi"),
    ("am I obese or normal", "bmi"),
    ("what is my body mass index", "bmi"),
    ("am I at a healthy weight", "bmi"),
    ("bmi check for 70kg and 170cm", "bmi"),
    # ── water_intake ──────────────────────────────────────────────────────────
    ("how much water should I drink", "water_intake"),
    ("daily water intake recommendation", "water_intake"),
    ("hydration goal for my weight", "water_intake"),
    ("how many litres of water per day", "water_intake"),
    ("water intake for fat loss", "water_intake"),
    ("how many glasses of water daily", "water_intake"),
    # ── snacks ────────────────────────────────────────────────────────────────
    ("suggest healthy snack ideas", "snacks"),
    ("what to eat between meals", "snacks"),
    ("give me snack suggestions", "snacks"),
    ("healthy evening snack ideas", "snacks"),
    ("pre workout snack ideas", "snacks"),
    ("post workout snack", "snacks"),
    ("healthy snacks for my diet", "snacks"),
    # ── tips ──────────────────────────────────────────────────────────────────
    ("give me diet tips", "tips"),
    ("nutrition advice for beginners", "tips"),
    ("healthy eating tips", "tips"),
    ("tips to lose weight faster", "tips"),
    ("diet hacks that actually work", "tips"),
    ("best practices for healthy eating", "tips"),
]

training_sentences, training_labels = zip(*_RAW)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

processed_sentences = [preprocess(s) for s in training_sentences]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_train = vectorizer.fit_transform(processed_sentences)

model = LogisticRegression(max_iter=500, C=2.0, random_state=42)
model.fit(X_train, training_labels)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STORE
# ═══════════════════════════════════════════════════════════════════════════════

sessions: dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "weight": None,    # kg
            "height": None,    # cm
            "age": None,
            "gender": None,    # "male" | "female"
            "activity": "moderate",
            "goal": None,      # fat_loss | muscle_gain | maintenance
            "diet_type": "veg",
        }
    return sessions[session_id]


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_weight(text: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)\s?kg", text.lower())
    return float(m.group(1)) if m else None


def extract_height(text: str) -> float | None:
    """Supports: 170cm  |  5'7"  |  5 feet 7  |  5 ft 7."""
    m = re.search(r"(\d+\.?\d*)\s?cm", text.lower())
    if m:
        return float(m.group(1))
    # feet + inches
    m2 = re.search(r"(\d+)\s*(?:feet|ft|')[\s,]*(\d+)", text.lower())
    if m2:
        return round(int(m2.group(1)) * 30.48 + int(m2.group(2)) * 2.54, 1)
    return None


def extract_age(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:year|yr|y/o|years old|age)", text.lower())
    return int(m.group(1)) if m else None


def extract_gender(text: str) -> str | None:
    lower = text.lower()
    if any(w in lower for w in ["male", " man ", "boy", " he ", " mr ", "i am a man"]):
        return "male"
    if any(w in lower for w in ["female", "woman", "girl", " she ", " ms ", " mrs "]):
        return "female"
    return None


def extract_activity(text: str) -> str | None:
    lower = text.lower()
    if any(w in lower for w in ["sedentary", "desk job", "no exercise", "inactive"]):
        return "sedentary"
    if any(w in lower for w in ["light", "yoga", "walk", "1-2 day"]):
        return "light"
    if any(w in lower for w in ["moderate", "3-4 day", "gym 3", "gym 4"]):
        return "moderate"
    if any(w in lower for w in ["very active", "intense", "athlete", "5-6 day", "6 day"]):
        return "very_active"
    if any(w in lower for w in ["active", "gym 5", "5 day"]):
        return "active"
    return None


def extract_goal(text: str) -> str | None:
    lower = text.lower()
    if any(w in lower for w in ["lose", "fat", "slim", "burn", "cut", "reduce", "deficit", "lean"]):
        return "fat_loss"
    if any(w in lower for w in ["gain", "muscle", "bulk", "mass", "build", "bigger", "grow", "surplus"]):
        return "muscle_gain"
    if any(w in lower for w in ["maintain", "maintenance", "stable", "keep"]):
        return "maintenance"
    return None


def extract_diet_type(text: str) -> str | None:
    lower = text.lower()
    if any(w in lower for w in ["non veg", "chicken", "egg", "fish", "meat", "beef", "seafood", "turkey", "tuna", "salmon", "prawn"]):
        return "non_veg"
    if any(w in lower for w in ["veg", "vegetarian", "plant", "vegan"]):
        return "veg"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVITY_MULTIPLIERS = {
    "sedentary":  1.200,
    "light":      1.375,
    "moderate":   1.550,
    "active":     1.725,
    "very_active": 1.900,
}


def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
    """Revised Harris-Benedict (1984)."""
    if gender == "female":
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    # male (default)
    return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)


def calculate_tdee(
    weight: float,
    height: float = 170,
    age: int = 25,
    gender: str = "male",
    activity: str = "moderate",
) -> int:
    bmr = calculate_bmr(weight, height, age, gender)
    return round(bmr * _ACTIVITY_MULTIPLIERS.get(activity, 1.55))


def calculate_target_calories(tdee: int, goal: str) -> int:
    if goal == "fat_loss":
        return tdee - 500
    if goal == "muscle_gain":
        return tdee + 300
    return tdee  # maintenance


def calculate_macros(weight: float, calories: int, goal: str) -> dict:
    """
    Protein targets per goal (g/kg):
      muscle_gain → 2.2 g/kg   fat_loss → 1.8 g/kg   maintenance → 1.6 g/kg
    Fat: 0.8 g/kg body weight.
    Remainder goes to carbs.
    """
    p_per_kg = {"muscle_gain": 2.2, "fat_loss": 1.8, "maintenance": 1.6}
    protein = round(weight * p_per_kg.get(goal, 1.8))
    fat = round(weight * 0.8)
    carbs = max(0, round((calories - protein * 4 - fat * 9) / 4))
    return {"protein": protein, "fat": fat, "carbs": carbs}


def calculate_bmi(weight: float, height: float) -> dict:
    """Returns BMI value and WHO category."""
    h_m = height / 100
    bmi = round(weight / (h_m ** 2), 1)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": bmi, "category": category}


def calculate_water(weight: float, activity: str = "moderate") -> float:
    """35 ml/kg body weight + 500 ml bonus for high-activity levels."""
    litres = round(weight * 35 / 1000, 1)
    if activity in ("active", "very_active"):
        litres = round(litres + 0.5, 1)
    return litres


# ═══════════════════════════════════════════════════════════════════════════════
# MEAL DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

_MEALS = {
    "veg": {
        "breakfast": [
            "Oats porridge + banana + almond milk",
            "Poha with peanuts + green tea",
            "Upma with mixed vegetables + coconut chutney",
            "Greek yogurt + mixed berries + granola",
            "Moong dal chilla × 3 + mint chutney",
            "Whole wheat toast + peanut butter + sliced apple",
            "Spinach-banana protein smoothie bowl",
        ],
        "lunch": [
            "Brown rice + dal + mixed sabzi + cucumber salad",
            "Chapati × 3 + paneer curry + curd",
            "Quinoa + roasted vegetables + tahini dressing",
            "Veg pulao + raita + pickle",
            "Rajma + 2 chapati + onion salad",
            "Lentil soup + whole grain bread + side salad",
            "Tofu stir-fry + brown rice + steamed broccoli",
        ],
        "dinner": [
            "Chapati × 2 + palak paneer + dal",
            "Vegetable khichdi + curd",
            "Tofu scramble + whole wheat roti + cucumber salad",
            "Mixed dal + rice + roasted veggies",
            "Paneer bhurji + 2 chapati + salad",
            "Veg soup + multigrain toast",
            "Chickpea curry + brown rice",
        ],
        "snacks": [
            "Handful of mixed nuts (almonds, walnuts, cashews)",
            "Apple + 2 tbsp peanut butter",
            "Roasted makhana (fox nuts) 30 g",
            "Hummus + carrot & cucumber sticks",
            "Banana + handful of almonds",
            "Greek yogurt + chia seeds + honey",
            "Sprouts chaat with lemon",
        ],
    },
    "non_veg": {
        "breakfast": [
            "3-egg omelette + whole wheat toast + orange juice",
            "3 boiled eggs + banana + black coffee",
            "Chicken sausage + scrambled eggs + multigrain toast",
            "Egg bhurji + multigrain paratha",
            "Greek yogurt + 2 boiled eggs + seasonal fruit",
            "Protein pancakes (egg + banana) + honey drizzle",
            "Tuna on whole wheat toast + black coffee",
        ],
        "lunch": [
            "Grilled chicken breast 150 g + brown rice + salad",
            "Chapati × 3 + chicken curry + raita",
            "Fish curry + white rice + mixed vegetable",
            "Chicken biryani (brown rice) + mint raita",
            "Prawn stir-fry + quinoa + steamed greens",
            "Chicken wrap (whole wheat) + Greek yogurt dip",
            "Salmon + sweet potato + asparagus",
        ],
        "dinner": [
            "Baked chicken 150 g + 2 chapati + green salad",
            "Grilled fish + roasted vegetables + brown rice",
            "Chicken clear soup + multigrain toast",
            "Egg curry + 2 chapati",
            "Stir-fried chicken + vegetables + noodles",
            "Turkey meatballs + whole wheat pasta + salad",
            "Shrimp fried rice (brown) + bok choy",
        ],
        "snacks": [
            "2 boiled eggs",
            "Canned tuna + whole grain crackers",
            "Chicken jerky 30 g",
            "Greek yogurt + honey + walnuts",
            "Protein shake + banana",
            "Grilled chicken strips + hummus dip",
            "Mini egg-white omelette",
        ],
    },
}


def generate_meal_plan(diet_type: str) -> dict:
    meals = _MEALS[diet_type]
    return {
        "breakfast": random.choice(meals["breakfast"]),
        "lunch":     random.choice(meals["lunch"]),
        "snacks":    random.choice(meals["snacks"]),
        "dinner":    random.choice(meals["dinner"]),
    }


def generate_weekly_plan(diet_type: str) -> dict:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return {day: generate_meal_plan(diet_type) for day in days}


# ═══════════════════════════════════════════════════════════════════════════════
# TIPS DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

_TIPS: dict[str, list[str]] = {
    "fat_loss": [
        "🔥 Eat in a 300–500 kcal/day deficit for sustainable fat loss (0.5 kg/week).",
        "💧 Drink a glass of water 20 min before meals to naturally reduce appetite.",
        "🌙 Stop eating 2–3 hours before bedtime to improve fat metabolism overnight.",
        "🏃 Add 30 min of cardio daily — even brisk walking counts.",
        "🍽️ Use smaller plates to control portion sizes without feeling deprived.",
        "📊 Track meals for 2–4 weeks — awareness alone reduces intake by ~10–15%.",
        "🥩 Prioritise protein — it keeps you full and preserves muscle while cutting.",
        "🚫 Eliminate liquid calories: sodas, juices, sweet coffee, alcohol.",
        "😴 Sleep 7–8 hrs — poor sleep spikes ghrelin (hunger hormone) by 24%.",
        "🧘 Manage stress — elevated cortisol causes preferential belly fat storage.",
    ],
    "muscle_gain": [
        "💪 Eat in a 200–300 kcal surplus for lean muscle gain without excess fat.",
        "🥩 Target 1.6–2.2 g of protein per kg of body weight every day.",
        "🏋️ Apply progressive overload — increase weight or reps by 2–5% each week.",
        "🍚 Don't fear carbs — they are the primary fuel for heavy training sessions.",
        "😴 Muscles repair and grow during sleep — prioritise 7–9 hours.",
        "⏱️ Space protein across 4–5 meals to maximise muscle protein synthesis.",
        "🥛 Have protein + fast carbs within 30 min post-workout for optimal recovery.",
        "📅 Commit to at least 12 weeks — visible muscle growth takes time.",
        "💧 Even 2% dehydration can reduce strength output by up to 10%.",
        "🩺 Creatine monohydrate (5 g/day) is the most evidence-backed supplement.",
    ],
    "maintenance": [
        "⚖️ Weigh yourself weekly (same time, fasted) to monitor stability.",
        "🥦 Focus on nutrient density — vitamins, minerals, fibre, and antioxidants.",
        "🍽️ Use the plate method: ½ vegetables, ¼ lean protein, ¼ complex carbs.",
        "🍫 Allow 10–20% of calories for treats — makes the diet sustainable long-term.",
        "💪 Continue resistance training to maintain muscle mass as you age.",
        "📊 Reassess calories every 6–8 weeks as your body composition changes.",
    ],
    "general": [
        "🥗 Eat a rainbow of vegetables daily for broad micronutrient coverage.",
        "💧 Aim for at least 2–3 litres of water daily — more if active.",
        "🍳 Cook more meals at home — you control ingredients and portion sizes.",
        "📖 Read nutrition labels — hidden sugars are in many 'healthy' products.",
        "🥜 Include healthy fats daily: nuts, avocado, olive oil, fatty fish.",
        "🚶 Hit 8,000–10,000 steps daily for baseline metabolic health.",
    ],
}

_GREETINGS = [
    (
        "👋 Hey there! I'm NutriBot, your AI diet assistant.\n\n"
        "Share your details for a personalised plan — e.g.:\n"
        "• *\"I'm 25M, 75 kg, 175 cm, moderate activity, want to lose fat\"*\n\n"
        "Or ask about: BMI · calories · protein · water intake · snacks · tips"
    ),
    "Hi! 🌿 Ready to build your perfect nutrition plan?\nShare your weight, height, age, and goal to get started!",
    "Hello! 🥗 I can help with meal plans, calorie targets, BMI, macros, hydration and more.\nWhat would you like to know?",
]

_GOODBYES = [
    "Stay consistent — results take time! 💪 See you!",
    "Great chatting! Remember: nutrition is a marathon, not a sprint. 🏃",
    "Good luck on your journey! Come back anytime. 🥗",
]

_FALLBACKS = [
    (
        "I'm not quite sure about that. Try asking me:\n"
        "• *\"75 kg fat loss veg diet plan\"*\n"
        "• *\"What is my BMI? I'm 70 kg and 170 cm\"*\n"
        "• *\"How much water should I drink?\"*\n"
        "• *\"Healthy snack ideas\"*\n"
        "• *\"Diet tips for muscle gain\"*"
    ),
    (
        "Hmm, I didn't catch that 🤔  You can ask me things like:\n"
        "• Diet / meal plans  • Calorie & macro targets\n"
        "• BMI check  • Protein intake\n"
        "• Water intake  • Healthy snacks  • Diet tips"
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RESPONSE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def chatbot_response(user_input: str, session_id: str) -> dict:
    profile = get_session(session_id)

    # ── Extract entities and update profile ──────────────────────────────────
    updates = {
        "weight":    extract_weight(user_input),
        "height":    extract_height(user_input),
        "age":       extract_age(user_input),
        "gender":    extract_gender(user_input),
        "activity":  extract_activity(user_input),
        "goal":      extract_goal(user_input),
        "diet_type": extract_diet_type(user_input),
    }
    for key, val in updates.items():
        if val is not None:
            profile[key] = val

    # ── Resolve working values (profile → sensible defaults) ─────────────────
    weight    = profile["weight"]    or 70.0
    height    = profile["height"]    or 170.0
    age       = profile["age"]       or 25
    gender    = profile["gender"]    or "male"
    activity  = profile["activity"]  or "moderate"
    goal      = profile["goal"]      or "fat_loss"
    diet_type = profile["diet_type"] or "veg"

    # ── Classify intent ──────────────────────────────────────────────────────
    processed   = preprocess(user_input)
    x           = vectorizer.transform([processed])
    intent      = model.predict(x)[0]
    confidence  = float(max(model.predict_proba(x)[0]))

    # ── Intent handlers ──────────────────────────────────────────────────────

    if intent == "greeting":
        return {"type": "text", "message": random.choice(_GREETINGS)}

    if intent == "goodbye":
        return {"type": "text", "message": random.choice(_GOODBYES)}

    if intent == "bmi":
        bmi_data = calculate_bmi(weight, height)
        return {
            "type":     "bmi",
            "weight":   weight,
            "height":   height,
            **bmi_data,
        }

    if intent == "water_intake":
        litres  = calculate_water(weight, activity)
        glasses = round(litres / 0.25)      # 250 ml per glass
        return {
            "type":     "water",
            "weight":   weight,
            "activity": activity,
            "litres":   litres,
            "glasses":  glasses,
        }

    if intent == "snacks":
        pool = _MEALS[diet_type]["snacks"]
        return {
            "type":      "snacks",
            "diet_type": diet_type,
            "snacks":    random.sample(pool, min(4, len(pool))),
        }

    if intent == "tips":
        tip_pool = _TIPS.get(goal, []) + _TIPS["general"]
        return {
            "type":  "tips",
            "goal":  goal,
            "tips":  random.sample(tip_pool, min(5, len(tip_pool))),
        }

    if intent == "protein":
        protein = round(weight * 2.0)
        return {
            "type":    "protein",
            "weight":  weight,
            "protein": protein,
            "tip":     "Spread your protein across 4–5 meals for optimal muscle protein synthesis.",
        }

    if intent == "calories":
        tdee   = calculate_tdee(weight, height, age, gender, activity)
        target = calculate_target_calories(tdee, goal)
        return {
            "type":     "calories",
            "weight":   weight,
            "goal":     goal,
            "tdee":     tdee,
            "calories": target,
        }

    if intent in ("diet_plan", "fat_loss", "muscle_gain", "maintenance"):
        # Map intent → goal if goal was not extracted from raw text
        if updates["goal"] is None:
            if intent in ("fat_loss", "muscle_gain", "maintenance"):
                goal = intent
            profile["goal"] = goal

        tdee   = calculate_tdee(weight, height, age, gender, activity)
        target = calculate_target_calories(tdee, goal)
        macros = calculate_macros(weight, target, goal)
        meals  = generate_meal_plan(diet_type)

        return {
            "type":      "diet_plan",
            "goal":      goal,
            "weight":    weight,
            "height":    height,
            "age":       age,
            "gender":    gender,
            "activity":  activity,
            "diet_type": diet_type,
            "tdee":      tdee,
            "calories":  target,
            "macros":    macros,
            "meal_plan": meals,
        }

    # ── Low-confidence fallback ───────────────────────────────────────────────
    if confidence < 0.40:
        return {"type": "text", "message": random.choice(_FALLBACKS)}

    return {"type": "text", "message": random.choice(_FALLBACKS)}


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""


@app.get("/")
def root():
    return {"message": "NutriBot API is running 🥗", "version": "2.0"}


@app.post("/chat")
def chat(req: ChatRequest):
    """Main conversation endpoint. Returns typed response dicts."""
    sid      = req.session_id or str(uuid.uuid4())
    response = chatbot_response(req.message, sid)
    response["session_id"] = sid
    return response


@app.post("/weekly-plan")
def weekly_plan(req: ChatRequest):
    """Generate a 7-day meal plan for the session's diet type."""
    sid       = req.session_id or str(uuid.uuid4())
    profile   = get_session(sid)
    diet_type = extract_diet_type(req.message) or profile.get("diet_type", "veg")
    plan      = generate_weekly_plan(diet_type)
    return {"type": "weekly_plan", "diet_type": diet_type, "plan": plan, "session_id": sid}


@app.get("/session/{session_id}")
def get_profile(session_id: str):
    """Inspect the stored profile for a given session."""
    return sessions.get(session_id, {})
