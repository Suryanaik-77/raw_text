import os
from dotenv import load_dotenv
import openai
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
import json
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"LLM returned invalid JSON:\n{text}")

def extract_metadata(doc_text: str):
    prompt = f"""
You are a VLSI documentation classifier.

TASK:
Classify the document and extract metadata according to STRICT ORGANIZATIONAL RULES.

DOCUMENT TYPES:
- "tool" → EDA tool documentation (licensed manuals, user guides)
- "theory" → textbooks, academic, research material

STRICT OUTPUT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No extra keys
- Use "Unknown" if information is missing
- Follow the schema EXACTLY

METADATA SCHEMA (ALWAYS USE ALL FIELDS):

{{
  "domain": "...",
  "stage": "...",
  "type": "...",
  "version": "...",
  "vendor": "..."
}}

MANDATORY CLASSIFICATION RULES (DO NOT VIOLATE):

1️⃣ DOMAIN RULE:
- domain MUST ALWAYS be "Physical Design"
- NEVER use Synthesis, STA, Timing, or VLSI as domain

2️⃣ TYPE RULE:
- If document is an EDA manual → type = "tool"
- If document is a book / academic → type = "theory"

3️⃣ STAGE RULE:
- For TOOL documents:
  - stage = primary function of the tool
  - Examples: Synthesis, Placement, CTS, Routing, Signoff

- For THEORY documents:
  - stage = technical topic
  - Examples: Static Timing Analysis, Clocking, Low Power

4️⃣ VERSION RULE:
- If TOOL → extract tool version if present else "Unknown"
- If THEORY → version MUST be "NA"

5️⃣ VENDOR RULE:
- If TOOL → tool vendor name
- If THEORY → vendor MUST be "NA"

IMPORTANT:
- Do NOT invent stages
- Do NOT use Frontend / Backend / Theory as stage
- Follow organizational taxonomy strictly

DOCUMENT TEXT:
<<<
{doc_text[:6000]}
>>>
"""
    response = llm.invoke(prompt)
    return safe_json_parse(response.content)

## output Format
## {"domain" : "","stage": "", "vendor": "", "version": "", type:""}
