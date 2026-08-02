"""
Shared constants used across the InsightAI application.

Keeping these in one place avoids magic strings scattered across modules
and makes it easy to tune keyword lists / thresholds later.
"""

# --- Column detection keyword hints -----------------------------------------
# These are only *hints*. Actual dtype (from Pandas) is always the primary
# signal; keyword matching is a secondary heuristic used to label columns
# for KPI calculation (e.g. "which numeric column is revenue?").
REVENUE_KEYWORDS = ["revenue", "sales", "amount", "total_sales", "turnover", "income"]
COST_KEYWORDS = ["cost", "expense", "expenditure", "cogs"]
PROFIT_KEYWORDS = ["profit", "margin_amount", "net_income", "earnings"]
PRICE_KEYWORDS = ["price", "unit_price", "rate"]
UNITS_KEYWORDS = ["units", "quantity", "qty", "units_sold", "count_sold"]
DATE_KEYWORDS = ["date", "order_date", "transaction_date", "timestamp", "created_at", "invoice_date"]
ID_KEYWORDS = ["id", "_id", "order_id", "invoice_no", "transaction_id", "customer_id"]
REGION_KEYWORDS = ["region", "state", "city", "location", "territory", "zone"]
PRODUCT_KEYWORDS = ["product", "item", "sku", "goods"]
CATEGORY_KEYWORDS = ["category", "segment", "type", "class"]

# --- File handling ------------------------------------------------------------
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
MAX_FILE_SIZE_MB = 200  # Soft warning threshold, not a hard block
MAX_PREVIEW_ROWS = 10

# --- Data quality score weights (must sum to 100) -----------------------------
# Score = 100 - (missing_penalty + duplicate_penalty), floored at 0.
# missing_penalty  = missing_value_ratio  * MISSING_WEIGHT
# duplicate_penalty = duplicate_row_ratio * DUPLICATE_WEIGHT
MISSING_WEIGHT = 60
DUPLICATE_WEIGHT = 40

# --- Gemini configuration ------------------------------------------------------
# Kept isolated here so the model name can be updated in one place if Google
# changes free-tier availability. gemini-2.5-flash is, as of this writing,
# a broadly available free-tier text model with a generous context window.
GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.0-flash"
GEMINI_MAX_OUTPUT_TOKENS = 2048
GEMINI_TEMPERATURE = 0.3
GEMINI_REQUEST_TIMEOUT_SECONDS = 30

AI_UNAVAILABLE_MESSAGE = (
    "AI features are currently unavailable. Core analytics are still available."
)

AI_SYSTEM_GUARDRAILS = """You are InsightAI, a careful business data analyst assistant.
You are analyzing structured data supplied by the application.

Rules you must always follow:
- Use ONLY the information provided in the dataset context below. Never invent
  numbers, products, regions, dates, or trends that are not present in the context.
- Do not perform arithmetic yourself when a calculated value has already been
  supplied — quote the supplied value instead of recomputing it.
- Never claim causation from correlation. You may describe a pattern, but you
  must not assert a definite cause unless the data context explicitly states one.
- If the available information is insufficient to answer a question, explicitly
  say: "The uploaded dataset does not contain enough information to answer that."
- Be concise, business-appropriate, and avoid filler language.
"""

# --- Trend analysis ------------------------------------------------------------
TREND_SIGNIFICANT_CHANGE_PCT = 10.0  # % change considered "significant"

# --- Currency ------------------------------------------------------------------
# We never assume a currency. If a dataset doesn't specify one, numbers are
# shown unprefixed with thousands separators only.
