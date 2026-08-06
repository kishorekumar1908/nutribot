# NutriBot - AI Diet Chatbot

NutriBot is an AI-powered diet and nutrition assistant with a FastAPI backend and a single-file HTML/JS frontend. It classifies user intent with a TF-IDF + Logistic Regression model, extracts personal details (weight, height, age, gender, activity level, diet type, goal) from free-text messages, and returns personalised diet plans, calorie/macro targets, BMI, water intake, snack ideas, and diet tips.

## Features

- **NLP intent classification** — NLTK preprocessing (tokenise → lemmatise → stop-word removal) feeding a TF-IDF + Logistic Regression model across 12 intent classes.
- **Entity extraction** — pulls weight, height, age, gender, activity level, diet type, and goal straight out of natural language (e.g. *"I'm 25M, 75kg, 175cm, moderate activity, want to lose fat"*).
- **Session-based profiles** — an in-memory, UUID-keyed session store remembers your details across a conversation.
- **Accurate calculations** — Harris-Benedict BMR × activity multiplier for TDEE, WHO-classified BMI, and goal-specific macro splits (protein/fat/carbs).
- **Meal plans** — daily meal plan generator and a 7-day weekly plan endpoint, with separate veg and non-veg meal databases.
- **Extras** — water intake recommendations, healthy snack suggestions, and goal-specific diet tips.
- **Polished chat UI** — light/dark mode, quick-reply chips, typing indicator, animated result cards (diet plan, BMI gauge, water tracker, etc.), a weekly-plan modal, and plain-text export of plans.

## Tech Stack

| Layer      | Technology                                    |
|------------|------------------------------------------------|
| Backend    | FastAPI, scikit-learn (TF-IDF + Logistic Regression), NLTK |
| Frontend   | Vanilla HTML/CSS/JS (single file, no build step) |
| Container  | Docker, Docker Compose, nginx (serves the frontend) |

## Project Structure

```
nutribot/
├── backend/
│   ├── diet_chatbot.py     # FastAPI app, NLP model, calculation engine, API routes
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Backend container image
├── frontend/
│   └── index.html          # Chat UI (HTML/CSS/JS, no build step)
├── docker-compose.yml       # Runs backend + frontend together
├── .dockerignore
└── .gitignore
```

## Getting Started

### Option 1 — Docker (recommended)

```bash
docker compose up --build
```

- Backend API → http://localhost:8000
- Frontend UI → http://localhost:8080

### Option 2 — Run locally without Docker

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API
uvicorn diet_chatbot:app --reload
```

The API runs at `http://127.0.0.1:8000`. Then just open `index.html` in your browser — it's a static file and talks directly to the API (CORS is open for local development).

## API Reference

| Method | Endpoint             | Description                                      |
|--------|-----------------------|---------------------------------------------------|
| GET    | `/`                    | Health check                                       |
| POST   | `/chat`                | Main conversation endpoint — send `{ "message": "...", "session_id": "..." }` |
| POST   | `/weekly-plan`         | Generates a 7-day meal plan for the session's diet type |
| GET    | `/session/{session_id}` | Inspect the stored profile for a session          |

### Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I am 25M, 75kg, 175cm, moderate activity, want to lose fat", "session_id": ""}'
```

## Notes

- User profiles and chat sessions are stored **in-memory** — they reset whenever the backend restarts. There's no database in this version.
- The frontend's `API` constant in `index.html` is hardcoded to `http://127.0.0.1:8000` — update it if you deploy the backend elsewhere.
