"""
Vercel entry point — imports the Flask app from src/app.py.
Vercel looks for a module-level `app` object in this file.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app  # noqa: F401 — Vercel uses this name
