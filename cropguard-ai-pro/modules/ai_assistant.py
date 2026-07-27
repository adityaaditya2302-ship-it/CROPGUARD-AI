"""
CropGuard AI - AI Farm Assistant (Phase 2)
Multilingual agricultural expert powered by LLM (Gemini Pro / GPT-4o).
Answers farmer questions in natural language with domain-specific context.

Capabilities:
  - Disease identification and explanation
  - Treatment recommendations
  - Weather-based spray advice
  - Fertilizer recommendations
  - Market price queries
  - Government scheme information
  - Irrigation advice
  - Yield estimation
  - Voice input support (via Web Speech API on frontend)
"""
import os
import json
from datetime import datetime


# ── System prompt for the agricultural LLM ────────────────────────────────────
AGRI_SYSTEM_PROMPT = """You are CropGuard AI Assistant, an expert agricultural advisor specializing in Indian and global farming.

You have deep expertise in:
- Crop disease identification, treatment, and prevention
- Pest management (IPM - Integrated Pest Management)
- Soil health, fertilization, and nutrient management
- Irrigation scheduling and water management
- Organic and chemical treatment options with dosages
- Weather impact on crops and spray timing
- Yield prediction and harvest planning
- Government agricultural schemes (PM-KISAN, PMFBY, Kisan Credit Card, etc.)
- Market prices and mandi rates
- Drone and precision agriculture
- Sustainable and eco-friendly farming practices

You communicate in a farmer-friendly way:
- Use simple language, avoid excessive jargon
- Provide specific dosages and practical steps
- Always mention both organic AND chemical options
- Be culturally aware (Indian farming practices, local crop names)
- Support Hindi, Telugu, Tamil, Marathi, Punjabi, and English
- When uncertain, recommend consulting local Krishi Vigyan Kendra (KVK)

IMPORTANT RULES:
- Never recommend illegal pesticides
- Always advise pre-harvest intervals for chemical treatments
- Recommend consulting local extension officers for rare diseases
- Prioritize farmer safety and environmental protection

Context will be provided about the farmer's current scan results, weather, and farm history.
"""

# ── Predefined quick responses for common questions (no API needed) ────────────
QUICK_RESPONSES = {
    "spray tomorrow":
        "To decide if you can spray tomorrow, I need today's weather forecast. "
        "Generally: avoid spraying if wind > 25 km/h, rain is expected, temperature > 35°C, or humidity > 90%. "
        "Best time to spray is early morning (6-10 AM) or late afternoon (4-6 PM).",

    "pm kisan":
        "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi) provides ₹6,000/year to eligible farmers "
        "in 3 installments of ₹2,000. Apply at pmkisan.gov.in or nearest Common Service Centre. "
        "Required documents: Aadhaar card, bank account, land records.",

    "crop insurance":
        "PMFBY (Pradhan Mantri Fasal Bima Yojana) provides crop insurance at very low premiums: "
        "Kharif: 2%, Rabi: 1.5%, Horticulture: 5%. Register through your bank, insurance company, "
        "or CSC before the deadline for each season.",

    "soil test":
        "Get a free or low-cost soil test at your nearest Krishi Vigyan Kendra (KVK) or "
        "Soil Testing Laboratory. The Soil Health Card scheme provides free testing every 2 years. "
        "Testing reveals N, P, K levels, pH, and micronutrient deficiencies.",

    "neem oil":
        "Neem oil is an excellent organic pesticide/fungicide. Use 5ml/L water with 1ml liquid soap "
        "as emulsifier. Spray every 7-10 days on affected plants, covering both leaf surfaces. "
        "Effective against aphids, whitefly, mites, early-stage fungal diseases.",
}


class AIFarmAssistant:
    """
    Multilingual AI assistant for farmers.
    Uses Gemini Pro by default; falls back to smart pattern matching.
    """

    def __init__(self):
        self.gemini_key  = os.environ.get("GEMINI_API_KEY", "")
        self.openai_key  = os.environ.get("OPENAI_API_KEY", "")
        self.use_gemini  = bool(self.gemini_key)
        self.use_openai  = bool(self.openai_key) and not self.use_gemini
        self.conversation_history = []

        if self.use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    system_instruction=AGRI_SYSTEM_PROMPT,
                )
                self.gemini_chat  = self.gemini_model.start_chat(history=[])
                print("✅ AI Assistant: Gemini Pro ready")
            except Exception as e:
                print(f"⚠️  Gemini init failed: {e}")
                self.use_gemini = False

        elif self.use_openai:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
                print("✅ AI Assistant: GPT-4o ready")
            except Exception as e:
                print(f"⚠️  OpenAI init failed: {e}")
                self.use_openai = False

        if not self.use_gemini and not self.use_openai:
            print("⚠️  AI Assistant: No API key found. Using smart offline responses.")

    def ask(self, question: str, context: dict = None) -> dict:
        """
        Answer a farmer's question.

        Args:
            question: the farmer's question (any language)
            context:  optional dict with keys like:
                        crop, disease, weather, farm_history, scan_result

        Returns:
            dict with answer, source, suggestions, language_detected
        """
        # Build context string
        context_str = self._build_context_string(context or {})
        full_prompt  = f"{context_str}\n\nFarmer's question: {question}" if context_str else question

        # Try quick response first (offline, instant)
        quick = self._check_quick_response(question)
        if quick:
            return {
                "answer":    quick,
                "source":    "offline_knowledge",
                "language":  "en",
                "follow_up": self._get_follow_up_suggestions(question),
            }

        # Try LLM
        if self.use_gemini:
            return self._ask_gemini(full_prompt, question)
        elif self.use_openai:
            return self._ask_openai(full_prompt, question)
        else:
            return self._ask_offline(question, context or {})

    def _ask_gemini(self, prompt: str, original_question: str) -> dict:
        try:
            response = self.gemini_chat.send_message(prompt)
            answer   = response.text
            self.conversation_history.append({"q": original_question, "a": answer})
            return {
                "answer":    answer,
                "source":    "gemini_pro",
                "language":  "auto_detected",
                "follow_up": self._get_follow_up_suggestions(original_question),
            }
        except Exception as e:
            print(f"⚠️  Gemini API error: {e}")
            return self._ask_offline(original_question, {})

    def _ask_openai(self, prompt: str, original_question: str) -> dict:
        try:
            messages = [
                {"role": "system",    "content": AGRI_SYSTEM_PROMPT},
                *[{"role": "user" if i % 2 == 0 else "assistant", "content": m}
                  for i, m in enumerate(self.conversation_history[-6:])],
                {"role": "user",      "content": prompt},
            ]
            resp   = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            answer = resp.choices[0].message.content
            self.conversation_history.append(original_question)
            self.conversation_history.append(answer)
            return {
                "answer":    answer,
                "source":    "gpt4o",
                "language":  "auto_detected",
                "follow_up": self._get_follow_up_suggestions(original_question),
            }
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}")
            return self._ask_offline(original_question, {})

    def _ask_offline(self, question: str, context: dict) -> dict:
        """Smart offline response using keyword matching + context."""
        q_lower = question.lower()
        answer  = "I can help with crop diseases, treatments, weather advice, and farming practices. "

        # Disease-related
        if any(w in q_lower for w in ["disease", "blight", "rust", "mold", "spot", "rot"]):
            answer += ("Use the 'Scan Crop' feature to upload a photo of your affected plant. "
                       "The AI will identify the exact disease and provide treatment. "
                       "Neem oil (5ml/L) is a safe first step for most fungal diseases.")

        # Irrigation
        elif any(w in q_lower for w in ["water", "irrigat", "drip", "moisture"]):
            answer += ("General irrigation guide: Most crops need 25-50mm water per week. "
                       "Use drip irrigation to reduce water use by 40%. "
                       "Water early morning to reduce evaporation. "
                       "Check soil moisture by pushing your finger 5cm deep - water if dry.")

        # Fertilizer
        elif any(w in q_lower for w in ["fertiliz", "nitrogen", "npk", "urea", "potash"]):
            answer += ("Without a soil test, a safe general schedule: "
                       "Apply NPK 10:26:26 @ 100kg/acre as basal dose. "
                       "Top dress with Urea (45% N) @ 50kg/acre at 30 and 60 days after transplanting. "
                       "For organic: Farm yard manure 4-5 tonnes/acre before sowing.")

        # Market price
        elif any(w in q_lower for w in ["price", "market", "mandi", "rate", "sell"]):
            answer += ("For live mandi prices: Visit agmarknet.gov.in or the eNAM (National Agriculture Market) portal. "
                       "In your mobile: Download the 'Kisan Suvidha' or 'Fasal' app for live prices. "
                       "The best time to sell is usually 2-4 weeks after peak harvest season.")

        # Spray timing
        elif any(w in q_lower for w in ["spray", "pesticide", "fungicide"]):
            answer += ("Best spray conditions: Wind < 15 km/h, no rain for 4 hours after spraying, "
                       "temperature 15-30°C, early morning or late afternoon. "
                       "Always wear gloves, mask, and protective clothing. "
                       "Follow pre-harvest interval (PHI) on the label.")

        else:
            answer += ("Please use the Crop Scanner for disease diagnosis, or ask me specifically about: "
                       "treatments, irrigation, fertilizer, weather, market prices, or government schemes.")

        return {
            "answer":    answer,
            "source":    "offline_smart",
            "language":  "en",
            "follow_up": self._get_follow_up_suggestions(question),
        }

    def _build_context_string(self, context: dict) -> str:
        parts = []
        if context.get("disease"):
            parts.append(f"Recent scan detected: {context['disease']} on {context.get('crop', 'unknown crop')}")
        if context.get("weather"):
            w = context["weather"]
            parts.append(f"Current weather: {w.get('temp', '?')}°C, Humidity: {w.get('humidity', '?')}%")
        if context.get("farm_name"):
            parts.append(f"Farm: {context['farm_name']}")
        return "Context: " + ". ".join(parts) if parts else ""

    def _check_quick_response(self, question: str) -> str | None:
        q_lower = question.lower()
        for trigger, response in QUICK_RESPONSES.items():
            if trigger in q_lower:
                return response
        return None

    def _get_follow_up_suggestions(self, question: str) -> list:
        q_lower = question.lower()
        if "disease" in q_lower or "blight" in q_lower:
            return [
                "What organic treatment can I use?",
                "How do I prevent it next season?",
                "How much yield loss to expect?",
            ]
        if "irrigat" in q_lower or "water" in q_lower:
            return [
                "How do I set up drip irrigation?",
                "What is soil moisture optimal level?",
                "Can water stress cause disease?",
            ]
        return [
            "How do I improve crop yield?",
            "What government schemes am I eligible for?",
            "When should I harvest?",
        ]

    def clear_history(self):
        self.conversation_history = []
        if self.use_gemini:
            try:
                self.gemini_chat = self.gemini_model.start_chat(history=[])
            except Exception:
                pass


# Singleton
_assistant_instance: AIFarmAssistant | None = None

def get_assistant() -> AIFarmAssistant:
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = AIFarmAssistant()
    return _assistant_instance
