from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.core.config import config

# Get 'app' config as ConfigNamespace, convert to dict for safe access
app_config = config.get("app")
app_config_dict = vars(app_config) if app_config else {}

# Get 'cors' config similarly
cors_config = config.get("cors")
cors_dict = vars(cors_config) if cors_config else {}

app = FastAPI(
    title=app_config_dict.get("name", "LexiDraft Pro"),
    version=app_config_dict.get("version", "0.1.0"),
    debug=app_config_dict.get("debug", False),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_dict.get("allowed_origins", []),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to LexiDraft Pro API"}
