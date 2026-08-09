import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


def gemini_connected():

    return API_KEY is not None


def generate_summary(context, query):

    if not API_KEY:

        return "❌ Gemini API key not found."

    prompt = f"""
You are InsightForge AI.

Answer professionally.

Question:
{query}

Context:
{context[:14000]}
"""

    try:

        model = genai.GenerativeModel(
	    "gemini-3.6-flash"
	)
        response = model.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return f"AI Error:\n{e}"