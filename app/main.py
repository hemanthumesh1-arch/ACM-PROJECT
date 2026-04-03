# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# IMPORTANT: Create app, add middleware, then include router — order matters
app = FastAPI(
    title="Autonomous Constellation Manager",
    description="NSH 2026 — Orbital Debris Avoidance & Constellation Management System",
    version="1.0.0",
)

# CORS must be added BEFORE routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve frontend as static files at root AFTER all API routes
# This catches everything not matched by API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
