"""
Intent Router Agent
===================
Classifies the user's message into a legal intent category.

Strategy (in order of priority):
  1. Keyword overrides  — fast, deterministic shortcuts for clear-cut intents
  2. Local fine-tuned model — loaded only when NOT running on Render and the
     model file is present and full-size (> 1 MB)
  3. Groq LLM fallback — always available; used when model is absent/on Render

The `RENDER=true` environment variable is set automatically by Render.com at
runtime, so this file requires zero code changes to switch between environments:
  - Local machine with model downloaded → uses local BERT classifier
  - Render deployment (or no model file) → uses Groq LLM
"""
import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
from utils.llm_utils import strip_think
from langchain_core.messages import HumanMessage

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models/intent_classifier")
MODEL_FILE = os.path.join(MODEL_DIR, "model.safetensors")
ONNX_QUANT_FILE = os.path.join(MODEL_DIR, "model_quantized.onnx")
ONNX_FILE = os.path.join(MODEL_DIR, "model.onnx")
MAPPING_FILE = os.path.join(MODEL_DIR, "label_mapping.json")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
_is_render = os.getenv("RENDER", "").lower() in ("true", "1", "yes")
_model_file_ok = (
    os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 1_000_000
)
_use_local_model = (not _is_render) and _model_file_ok

# ---------------------------------------------------------------------------
# 1. Load fine-tuned model (ONNX preferred for low RAM & high speed)
# ---------------------------------------------------------------------------
DEFAULT_LABEL_MAPPING = {
    0: "Cheque_Bounce",
    1: "Civic_Scheme_Info",
    2: "Consumer_Dispute",
    3: "Criminal_FIR",
    4: "Cybercrime",
    5: "Legal_Notice_Contract",
    6: "RERA_RealEstate",
    7: "RTI",
    8: "Tenant_Landlord",
    9: "Workplace_Labour",
}

onnx_session = None
onnx_tokenizer = None
local_model = None
local_tokenizer = None
label_mapping: dict[int, str] = DEFAULT_LABEL_MAPPING.copy()

# Always load the label mapping
if os.path.exists(MAPPING_FILE):
    try:
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            label_mapping.update({int(k): v for k, v in raw.items()})
    except Exception:
        pass

# Determine ONNX model path (prioritize quantized INT8: ~105MB, <150MB RAM)
onnx_model_path = None
if os.path.exists(ONNX_QUANT_FILE) and os.path.getsize(ONNX_QUANT_FILE) > 10_000_000:
    onnx_model_path = ONNX_QUANT_FILE
elif os.path.exists(ONNX_FILE) and os.path.getsize(ONNX_FILE) > 10_000_000:
    onnx_model_path = ONNX_FILE
elif _is_render:
    # On Render, auto-fetch quantized ONNX model from Hugging Face if not bundled
    try:
        import urllib.request
        hf_repo = os.getenv("HF_MODEL_ID", "EverVissionAI/jansaathi-legal-intent").replace("https://huggingface.co/", "").strip("/")
        hf_url = f"https://huggingface.co/{hf_repo}/resolve/main/model_quantized.onnx"
        print(f"[IntentRouter] Downloading quantized ONNX model from {hf_url}...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(hf_url, ONNX_QUANT_FILE)
        if os.path.exists(ONNX_QUANT_FILE) and os.path.getsize(ONNX_QUANT_FILE) > 10_000_000:
            onnx_model_path = ONNX_QUANT_FILE
            print("[IntentRouter] Quantized ONNX model downloaded successfully!")
    except Exception as exc:
        print(f"[IntentRouter] Could not download ONNX model on Render: {exc}")

if onnx_model_path and os.path.exists(TOKENIZER_FILE):
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        onnx_session = ort.InferenceSession(onnx_model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        onnx_tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
        print(f"[IntentRouter] ONNX model active ({os.path.basename(onnx_model_path)}). RAM < 150MB. Classes: {list(label_mapping.values())}")
    except Exception as exc:
        print(f"[IntentRouter] Failed to load ONNX model: {exc}")
        onnx_session = None
        onnx_tokenizer = None

# PyTorch fallback if ONNX not present and running locally
if onnx_session is None and _use_local_model:
    try:
        import torch  # noqa: imported conditionally to avoid OOM on Render
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print("[IntentRouter] Local environment detected. Loading PyTorch model...")
        local_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        local_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        local_model.eval()
        print(f"[IntentRouter] PyTorch model loaded. Classes: {list(label_mapping.values())}")
    except Exception as exc:
        print(f"[IntentRouter] Failed to load PyTorch model: {exc}. Falling back to Hugging Face / Groq.")
        local_model = None
        local_tokenizer = None
elif onnx_session is None:
    if _is_render:
        print("[IntentRouter] Render environment: ONNX not yet ready. Using Hugging Face / Groq fallback.")
    else:
        print("[IntentRouter] Local model not found. Using Hugging Face / Groq fallback.")

# ---------------------------------------------------------------------------
# 2. Groq LLM fallback (always initialised)
# ---------------------------------------------------------------------------
_llm = ChatGroq(
    model=os.getenv("MODEL_CHEAP", "openai/gpt-oss-20b"),
    temperature=0.0,
)

_intent_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert Indian legal intent classifier. "
        "Read the user's message IN THE CONTEXT of the recent conversation and classify "
        "their true intent into exactly ONE of these categories:\n"
        "'RTI', 'Consumer_Dispute', 'RERA_RealEstate', 'Criminal_FIR', "
        "'Cybercrime', 'Workplace_Labour', 'Tenant_Landlord', 'Cheque_Bounce', "
        "'Legal_Notice_Contract', 'Civic_Scheme_Info', or 'Chitchat'.\n"
        "Return ONLY the category string, nothing else.\n\n"
        "Recent Conversation History:\n{history}",
    ),
    ("user", "{message}"),
])

_fallback_chain = _intent_prompt | _llm


# ---------------------------------------------------------------------------
# Pre-Check Guardrails & Helper Sets
# ---------------------------------------------------------------------------
import re

# Multi-word legal phrases (safe for substring matching)
_LEGAL_PHRASES = {
    "police complaint", "zero fir", "charge sheet", "chargesheet",
    "anticipatory bail", "court notice", "eviction notice", "demand notice",
    "legal notice", "power of attorney", "stamp paper", "rent agreement",
    "security deposit", "rent dispute", "right to information", "first appeal",
    "second appeal", "possession delay", "completion certificate", "occupancy certificate",
    "consumer forum", "consumer court", "defective product", "refund refused",
    "unfair trade", "e-commerce fraud", "cheque bounce", "cheque bounced",
    "dishonoured cheque", "dishonored cheque", "138 ni act", "cyber fraud",
    "online fraud", "otp fraud", "upi fraud", "bank fraud", "sim swap",
    "identity theft", "morphed photo", "labour court", "labor court",
    "unpaid salary", "wrongful termination", "pf claim", "sexual harassment at workplace",
    "domestic violence", "child custody", "motor vehicles act", "accident claim",
    "insurance claim refused", "supreme court", "high court", "district court",
    "sessions court", "lok adalat", "police station", "public prosecutor",
    "legal aid", "fundamental right", "fundamental rights", "writ petition",
    "consumer dispute", "property dispute", "illegal possession", "cheating case",
    "fraud case", "defamation notice",
}

# Single-word legal keywords & statutory terms (matched on word boundaries)
_LEGAL_WORDS = {
    "section", "ipc", "crpc", "bns", "bnss", "bsa", "cpc", "statute", "constitution",
    "constitutional", "amendment", "writ", "petition", "pil", "fir", "bail",
    "summons", "warrant", "affidavit", "notary", "poa", "probate", "succession",
    "sub-registrar", "mutation", "khata", "challan", "rti", "pio", "cic", "sic",
    "rera", "ncdrc", "scdrc", "dcdrc", "warranty", "tenant", "landlord", "eviction",
    "cybercrime", "phishing", "ransomware", "gratuity", "epfo", "esi", "posh",
    "pocso", "ndps", "mact", "tribunal", "nclt", "nclat", "magistrate", "sho", "dsp",
    "advocate", "lawyer", "vakil", "dlsa", "nalsa", "498a", "pwdva", "alimony",
    "divorce", "maintenance", "defamation", "cheque", "license", "claim", "surveyor",
    "insurance", "development", "authority", "refund", "illegal", "criminal", "fraud",
    "scam", "agreement", "contract", "property", "builder", "developer", "penalty",
    "interest", "allottee", "buyer", "seller", "compensation", "dispute", "litigation",
    "municipal", "certificate", "scheme", "yojana", "sp"
}

# Multi-word & single-word civic / political entities (Indian leaders & bodies)
_CIVIC_POLITICAL_PHRASES = {
    "narendra modi", "rahul gandhi", "arvind kejriwal", "amit shah",
    "yogi adityanath", "mamata banerjee", "draupadi murmu", "jagdeep dhankhar",
    "dy chandrachud", "d.y. chandrachud", "chief justice of india", "attorney general",
    "solicitor general", "chief minister", "prime minister", "president of india",
    "vice president", "finance minister", "home minister", "law minister",
    "railway minister", "defense minister", "external affairs minister",
    "s jaishankar", "nirmala sitharaman", "nitin gadkari", "smriti irani",
    "sonia gandhi", "priyanka gandhi", "mallikarjun kharge", "akhilesh yadav",
    "tejaswi yadav", "sharad pawar", "uddhav thackeray", "mk stalin",
    "pinarayi vijayan", "nitish kumar", "hemant soren", "siddaramaiah",
    "revanth reddy", "himanta biswa sarma", "bhagwant mann",
    "election commission", "enforcement directorate", "reserve bank",
    "municipal corporation", "gram panchayat", "gram sabha", "electricity board",
    "jal board", "water board", "bharatiya janata party", "indian national congress",
    "aam aadmi party", "samajwadi party", "shiv sena", "india alliance",
}

_CIVIC_POLITICAL_WORDS = {
    "modi", "kejriwal", "yogi", "mamata", "chandrachud", "cji",
    "governor", "minister", "mp", "mla", "mayor", "parliament",
    "lok sabha", "rajya sabha", "vidhan sabha", "eci", "cbi", "ed",
    "rbi", "sebi", "nhrc", "cag", "upsc", "ssc", "uidai", "aadhaar",
    "pan card", "voter id", "passport", "mcd", "bmc", "dda",
    "panchayat", "bjp", "inc", "congress", "aap", "tmc", "bsp", "sp",
    "dmk", "aiadmk", "cpim", "cpi", "ncp", "jdu", "rjd", "nda",
}

# Greetings patterns
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|heyy|heya|hiya|howdy|greetings)[\s!.,?]*$",
    r"^good\s+(morning|afternoon|evening|night|day)[\s!.,?]*$",
    r"^(how are you|how do you do|how're you|how r u|what's up|whats up|wassup|wazzup)[\s!.,?]*$",
    r"^(namaste|namaskar|pranam|khammaghani|ram ram|radhe radhe|sasriyakaal|sat sri akaal|adaab|salaam|salam)[\s!.,?]*$",
    r"^(kaise ho|kya haal hai|kya hal hai|kaise hain|kya chal raha hai|sab theek|sab kaisa hai)[\s!.,?]*$",
    r"^(thanks|thank you|thank u|thx|dhanyawad|shukriya|bahut shukriya|thanks a lot)[\s!.,?]*$",
    r"^(ok|okay|k|okk|theek hai|thik hai|sahi hai|alright|got it|understood|accha|acha)[\s!.,?]*$",
    r"^(bye|goodbye|see you|tata|alvida|cya)[\s!.,?]*$",
    r"^(yes|no|haan|nahi|nah|yep|nope|sure)[\s!.,?]*$",
]

_BOT_INTRO_PATTERNS = [
    "who are you", "what are you", "what can you do", "what do you do",
    "introduce yourself", "tell me about yourself", "who made you", "who created you",
    "aap kaun ho", "tum kaun ho", "aap kya karte ho", "tum kya karte ho",
    "kya kaam hai", "tumhara kaam", "aapka kaam", "kya help kar sakte",
    "kaise madad", "kya kar sakte ho", "help me", "can you help me",
    "kya ho tum", "kya hai tum", "help", "madad chahiye",
]

# Entertainment & Creative Requests
_CREATIVE_PATTERNS = [
    r"\b(joke|jokes|chutkula|chutkule|hasao|make me laugh|funny)\b",
    r"\b(story|stories|kahani|kisse|fairytale)\b",
    r"\b(poem|poetry|kavita|shayari|ghazal|rhyme)\b",
    r"\b(song|gaana|sing|lyrics|music)\b",
    r"\b(riddle|paheli|game|quiz|puzzle)\b",
]

_FOOD_KEYWORDS = {
    "recipe", "recipes", "how to cook", "how to make", "ingredients for", "kaise banaye",
    "kaise banate", "cooking", "bake", "fry", "biryani", "butter chicken", "paneer",
    "dal makhani", "samosa", "dosa", "idli", "chole bhature", "pizza", "pasta", "burger",
    "cake", "sandwich", "noodles", "maggi", "curry", "roti", "paratha", "naan",
    "chai", "tea", "coffee", "cocktail", "mocktail", "dessert", "ice cream",
}

_POP_CULTURE_KEYWORDS = {
    "spiderman", "spider-man", "spider man", "peter parker", "batman", "bruce wayne", "superman",
    "ironman", "iron man", "tony stark", "captain america", "thor", "hulk",
    "avengers", "marvel", "mcu", "dc comics", "justice league", "joker", "thanos",
    "deadpool", "wolverine", "x-men", "black panther", "doctor strange",
    "harry potter", "voldemort", "dumbledore", "hogwarts", "star wars", "darth vader",
    "luke skywalker", "yoda", "lord of the rings", "frodo", "gandalf", "game of thrones",
    "anime", "naruto", "sasuke", "goku", "dragon ball", "luffy", "one piece", "zoro",
    "death note", "attack on titan", "pokemon", "pikachu", "cartoon", "mickey mouse",
    "shinchan", "doraemon", "ben 10", "barbie", "disney", "pixar",
    "video game", "gaming", "playstation", "xbox", "nintendo", "gta", "grand theft auto",
    "minecraft", "fortnite", "pubg", "free fire", "call of duty", "valorant", "fifa",
    "shah rukh khan", "srk", "salman khan", "amitabh bachchan", "ranbir kapoor",
    "ranveer singh", "deepika padukone", "alia bhatt", "katrina kaif", "kareena kapoor",
    "taylor swift", "justin bieber", "selena gomez", "ariana grande", "beyonce",
    "drake", "eminem", "bts", "blackpink",
    "lionel messi", "messi", "cristiano ronaldo", "ronaldo", "neymar", "mbappe",
    "virat kohli", "rohit sharma", "ms dhoni", "dhoni", "sachin tendulkar",
    "ipl", "world cup score", "match score", "bollywood", "hollywood", "netflix",
}

_STEM_KEYWORDS = {
    "photosynthesis", "quantum physics", "quantum mechanics", "theory of relativity",
    "speed of light", "distance to sun", "distance to moon", "solar system", "planets",
    "black hole", "mitochondria", "dna structure", "periodic table", "atomic mass",
    "pythagoras theorem", "calculus", "differential equation", "algebra", "quadratic equation",
    "write python code", "write java code", "debug code", "javascript tutorial", "html css",
    "c++ program", "binary tree", "capital of france", "capital of usa", "capital of germany",
    "tallest mountain", "longest river", "who invented telephone", "who invented bulb",
    "dinosaurs", "jurassic", "world war 1", "world war 2",
}

_LEGAL_PHRASES_REGEX = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _LEGAL_PHRASES) + r')\b', re.IGNORECASE)
_CIVIC_POLITICAL_PHRASES_REGEX = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _CIVIC_POLITICAL_PHRASES) + r')\b', re.IGNORECASE)
_BOT_INTRO_REGEX = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _BOT_INTRO_PATTERNS) + r')\b', re.IGNORECASE)
_FOOD_REGEX = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _FOOD_KEYWORDS) + r')\b', re.IGNORECASE)
_POP_CULTURE_REGEX = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _POP_CULTURE_KEYWORDS) + r')\b', re.IGNORECASE)
_STEM_REGEX = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _STEM_KEYWORDS) + r')\b', re.IGNORECASE)


def _has_legal_anchors(text_lower: str) -> bool:
    """Returns True if query contains genuine legal terms or statutory anchors."""
    if _LEGAL_PHRASES_REGEX.search(text_lower):
        return True
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", text_lower))
    return bool(tokens.intersection(_LEGAL_WORDS))


def _has_civic_figures(text_lower: str) -> bool:
    """Returns True if query mentions Indian civic/political leaders or government bodies."""
    if _CIVIC_POLITICAL_PHRASES_REGEX.search(text_lower):
        return True
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", text_lower))
    return bool(tokens.intersection(_CIVIC_POLITICAL_WORDS))


def _precheck_intent(message: str) -> dict | None:
    """
    Deterministic pre-check guardrail that intercepts chitchat, greetings, jokes,
    recipes, pop culture, STEM trivia, and non-legal 'who is/what is' queries BEFORE
    the 10-class legal ML classifier or Groq fallback runs.
    
    Returns:
        dict: {"user_intent": intent, "next_action": "general_chat"} if caught
        None: if the query should proceed to legal classification
    """
    if not message or not message.strip():
        return {"user_intent": "Chitchat", "next_action": "general_chat"}

    msg_raw = message.strip()
    msg_lower = msg_raw.lower()
    
    # Check if query contains real legal dispute anchors
    has_legal = _has_legal_anchors(msg_lower)

    # 1. Greetings & Conversational Affirmations
    if not has_legal:
        for pat in _GREETING_PATTERNS:
            if re.search(pat, msg_lower, re.IGNORECASE):
                print(f"[IntentRouter PreCheck] Greeting matched: '{msg_raw[:50]}'")
                return {"user_intent": "Chitchat", "next_action": "general_chat"}

    # 2. Bot Identity / Intro Queries
    if not has_legal:
        if _BOT_INTRO_REGEX.search(msg_lower):
            print(f"[IntentRouter PreCheck] Bot intro matched: '{msg_raw[:50]}'")
            return {"user_intent": "Chitchat", "next_action": "general_chat"}

    # 3. Creative / Entertainment Requests (Jokes, Stories, Poems, Music)
    if not has_legal:
        for pat in _CREATIVE_PATTERNS:
            if re.search(pat, msg_lower, re.IGNORECASE):
                print(f"[IntentRouter PreCheck] Creative/Joke matched: '{msg_raw[:50]}'")
                return {"user_intent": "Off-Topic", "next_action": "general_chat"}

    # 4. Food & Recipes
    if not has_legal:
        if _FOOD_REGEX.search(msg_lower):
            print(f"[IntentRouter PreCheck] Food/Recipe matched: '{msg_raw[:50]}'")
            return {"user_intent": "Off-Topic", "next_action": "general_chat"}

    # 5. Pop Culture, Superheroes, Celebrities, Sports, Fiction
    if not has_legal:
        if _POP_CULTURE_REGEX.search(msg_lower):
            print(f"[IntentRouter PreCheck] Pop culture/Celebrity matched: '{msg_raw[:50]}'")
            return {"user_intent": "Off-Topic", "next_action": "general_chat"}

    # 6. STEM / Pure Science / Math / Tech / General Trivia
    if not has_legal:
        if _STEM_REGEX.search(msg_lower):
            print(f"[IntentRouter PreCheck] STEM/Trivia matched: '{msg_raw[:50]}'")
            return {"user_intent": "Off-Topic", "next_action": "general_chat"}

    # 7. "Who is / What is / Tell me about" General Questions
    _who_what_prefixes = [
        "who is", "who was", "who are", "who's", "whose",
        "what is", "what was", "what are", "what's",
        "tell me about", "explain about", "information about",
        "kaun hai", "kaun the", "kaun hain", "kon hai",
        "kya hai", "kya tha", "kya hote hain", "kya hota hai",
        "ke baare mein batao", "ke bare mein batao", "ke baare me batao", "ke bare me batao",
        "ke baare mein", "ke bare mein", "batao", "bataiye",
    ]
    _who_what_regex = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in _who_what_prefixes) + r')\b', re.IGNORECASE)
    
    is_who_what_query = bool(_who_what_regex.search(msg_lower))
    
    if is_who_what_query:
        # A) Indian political leaders, ministers, public institutions -> general_chat handles civics directly
        if _has_civic_figures(msg_lower) and not has_legal:
            print(f"[IntentRouter PreCheck] Civic/Political Figure inquiry: '{msg_raw[:50]}' -> General Chat")
            return {"user_intent": "Chitchat", "next_action": "general_chat"}
        
        # B) Other general who/what queries without legal anchors
        if not has_legal:
            print(f"[IntentRouter PreCheck] Non-legal 'who is/what is' query: '{msg_raw[:50]}' -> General Chat")
            return {"user_intent": "Off-Topic", "next_action": "general_chat"}

    return None


def _keyword_override(msg_lower: str) -> str | None:
    """
    Fast keyword-based shortcuts that bypass the ML model entirely.
    Returns an intent string or None if no override applies.
    """
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", msg_lower))
    
    # Document filling
    if "fill" in tokens and tokens.intersection({"document", "rti", "notice", "form", "it"}):
        return "Fill_Document"

    # Cheque bounce
    _cheque_bounce_kws = {
        "cheque bounce", "cheque bounced", "check bounce",
        "dishonoured cheque", "dishonored cheque", "bounced cheque",
        "negotiable instruments",
    }
    if any(kw in msg_lower for kw in _cheque_bounce_kws): # Phrases are fine here since they are multi-word and distinct
        return "Cheque_Bounce"

    # Domestic violence
    _domestic_violence_kws = {
        "domestic violence", "husband beats", "husband hits",
        "marital abuse", "wife beating", "498a", "498-a", "pwdva",
    }
    if any(kw in msg_lower for kw in _domestic_violence_kws) or tokens.intersection({"498a", "pwdva"}):
        return "Domestic_Violence"

    # Consumer complaints
    _consumer_kws = {
        "defective product", "consumer forum", "refund refused", "online shopping"
    }
    if any(kw in msg_lower for kw in _consumer_kws) or tokens.intersection({"bought", "purchased", "warranty", "e-commerce", "amazon", "flipkart"}):
        return "Consumer_Dispute"

    # Cybercrime
    _cyber_kws = {
        "online fraud", "cyber fraud", "otp fraud", "upi fraud", "morphed photo"
    }
    if any(kw in msg_lower for kw in _cyber_kws) or tokens.intersection({"phishing", "hacked", "cybercrime", "ransomware"}):
        return "Cybercrime"

    return None


# Valid intents for normalisation
_VALID_INTENTS = [
    "RTI", "Consumer_Dispute", "RERA_RealEstate", "Criminal_FIR",
    "Cybercrime", "Workplace_Labour", "Tenant_Landlord", "Cheque_Bounce",
    "Legal_Notice_Contract", "Civic_Scheme_Info", "Chitchat", "Off_Topic",
]

_DRAFTING_INTENTS = {
    "RTI", "Criminal_FIR", "Legal_Notice_Contract",
}

_ADVICE_INTENTS = {
    "Consumer_Dispute", "RERA_RealEstate", "Cybercrime",
    "Workplace_Labour", "Tenant_Landlord", "Cheque_Bounce",
    "Civic_Scheme_Info",
}


def intent_router_node(state: AgentState) -> dict:
    """
    LangGraph node: classifies user intent and sets next_action.

    Returns a dict with:
      - user_intent: one of the _VALID_INTENTS
      - next_action: 'draft_document' | 'retrieve_context' | 'general_chat'
    """
    messages = state.get("messages", [])
    if not messages:
        return {"user_intent": "General_Legal_Advice", "next_action": "general_chat"}

    latest_message = messages[-1].content
    msg_lower = latest_message.lower()

    # ── 1. PRE-CHECK GUARDRAIL (Zero-latency, rule-based) ───────────────────
    # Catches chitchat, off-topic, jokes, recipes, pop culture, and general knowledge
    # without sending to the 10-class legal ML classifier.
    precheck_res = _precheck_intent(latest_message)
    if precheck_res:
        print(f"[IntentRouter] Pre-check intercepted -> Intent: {precheck_res['user_intent']} | Next: {precheck_res['next_action']}")
        return precheck_res

    # ── 2. Keyword overrides (deterministic, model-independent) ─────────────
    intent = _keyword_override(msg_lower)
    if intent:
        print(f"[IntentRouter] Keyword override -> {intent}")

    # ── ONNX model inference (Render & Local, ultra-fast & low memory) ─────────
    if intent is None and onnx_session is not None and onnx_tokenizer is not None:
        try:
            import numpy as np
            enc = onnx_tokenizer.encode(latest_message)
            input_ids = np.array([enc.ids], dtype=np.int64)
            attention_mask = np.array([enc.attention_mask], dtype=np.int64)
            ort_outs = onnx_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            logits = ort_outs[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            pred_id = int(np.argmax(probs, axis=-1)[0])
            confidence = float(probs[0][pred_id])

            if confidence > 0.40:
                intent = label_mapping.get(pred_id)
                print(
                    f"[IntentRouter] ONNX model -> {intent} "
                    f"(confidence={confidence:.2f})"
                )
            else:
                print(
                    f"[IntentRouter] ONNX model uncertain "
                    f"(confidence={confidence:.2f}) -> Groq fallback."
                )
        except Exception as exc:
            print(f"[IntentRouter] ONNX inference error: {exc} -> Groq fallback.")
            intent = None

    # ── PyTorch model inference (local environment only) ───────────────────────
    if intent is None and local_model is not None:
        try:
            import torch
            inputs = local_tokenizer(
                latest_message,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = local_model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_id = torch.max(probs, dim=-1)

            # Threshold at 0.40 — 19-class model is confident at this level
            if confidence.item() > 0.40:
                intent = label_mapping.get(predicted_id.item())
                print(
                    f"[IntentRouter] Local model -> {intent} "
                    f"(confidence={confidence.item():.2f})"
                )
            else:
                print(
                    f"[IntentRouter] Local model uncertain "
                    f"(confidence={confidence.item():.2f}) -> Groq fallback."
                )
        except Exception as exc:
            print(f"[IntentRouter] Local inference error: {exc} -> Groq fallback.")
            intent = None

    # ── Hugging Face API fallback ──────────────────────────────────────────
    if intent is None:
        hf_model_id = os.getenv("HF_MODEL_ID")
        if hf_model_id:
            hf_model_id = hf_model_id.replace("https://huggingface.co/", "").strip("/")
        
        hf_token = os.getenv("HF_API_TOKEN")
        if hf_model_id and hf_token:
            from huggingface_hub import InferenceClient
            
            # Fix for huggingface_hub bug: internal API calls require the env var for private repos
            os.environ["HF_TOKEN"] = hf_token
            
            try:
                print(f"[IntentRouter] Attempting Hugging Face API fallback ({hf_model_id})...")
                client = InferenceClient(token=hf_token)
                # Call text classification API
                result = client.text_classification(latest_message, model=hf_model_id)
                # Result is a list of dicts: [{'label': 'LABEL_X', 'score': 0.9}]
                if isinstance(result, list) and len(result) > 0:
                    top_pred = result[0]
                    score = top_pred.get("score", 0.0)
                    if score > 0.40:
                        label_str = top_pred.get("label", "")
                        if label_str.startswith("LABEL_"):
                            label_id = int(label_str.replace("LABEL_", ""))
                            intent = label_mapping.get(label_id)
                        else:
                            intent = label_str
                        print(f"[IntentRouter] Hugging Face API -> {intent} (score={score:.2f})")
                    else:
                        print(f"[IntentRouter] Hugging Face API uncertain ({score:.2f})")
            except Exception as e:
                print(f"[IntentRouter] Hugging Face API request failed: {e}")

    # ── Groq LLM fail-safe (if HF fails or is not configured) ─────────────
    if intent is None:
        history_str = ""
        if len(messages) > 1:
            for m in messages[-5:-1]:
                role = "User" if isinstance(m, HumanMessage) else "JanSaathi"
                content = m.content[:300] + "..." if len(m.content) > 300 else m.content
                history_str += f"{role}: {content}\n"

        raw = _fallback_chain.invoke({"message": latest_message, "history": history_str})
        intent = strip_think(raw.content.strip())
        print(f"[IntentRouter] Groq fail-safe fallback -> {intent}")

    # ── Normalise to a valid intent ──────────────────────────────────────────
    intent_clean = "Chitchat"
    for valid in _VALID_INTENTS:
        if valid.lower() in str(intent).lower():
            intent_clean = valid
            break

    # ── Determine next action ────────────────────────────────────────────────
    if intent_clean in _DRAFTING_INTENTS:
        next_action = "draft_document"
    elif intent_clean in _ADVICE_INTENTS:
        next_action = "retrieve_context"
    else:
        next_action = "general_chat"

    print(f"[IntentRouter] FINAL -> Intent: {intent_clean} | Next: {next_action}")
    return {"user_intent": intent_clean, "next_action": next_action}
