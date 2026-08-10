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
You are InsightForge AI, a document question-answering assistant.

Answer the user's question using the provided document context.

Rules:
- Use the retrieved context as the primary source of information.
- If the question refers to a filename, identify the relevant "Source file" in the context.
- If the answer is present in the context, answer it directly.
- Do not say information is missing when it is present in the context.
- Do not invent or assume facts that are not present in the context.
- If the context genuinely does not contain the answer, say that the information was not found in the uploaded documents.
- Give a concise, clear answer.

Question:
{query}

Retrieved document context:
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