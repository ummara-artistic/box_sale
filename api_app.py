import os
import json
import re
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Generator

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load JSON data once at startup
import os
import json

file_path = os.path.join(os.getcwd(), "sales_data.json")

with open(file_path, "r", encoding="utf-8") as f:
    SALES_DATA = json.load(f)

# FastAPI app
app = FastAPI()


from datetime import datetime

def build_prompt(user_prompt: str, data=None) -> str:
    sample_data = ""
    latest_data_date = "unknown"

    if data:
        try:
            # Filter trx_date only for 2024 and 2025 records
            dates = [
                datetime.fromisoformat(item["trx_date"])
                for item in data.get("items", [])
                if "trx_date" in item and datetime.fromisoformat(item["trx_date"]).year in [2024, 2025]
            ]
            if dates:
                latest = max(dates)
                latest_data_date = latest.strftime("%B %Y")
        except Exception:
            latest_data_date = "[Could not extract date]"

        try:
            # Limit to ~8000 characters to prevent overload
            sample_data = f"\nHere is some sample data:\n{json.dumps(data, indent=2)[:8000]}"
        except Exception:
            sample_data = "\n[Sample data could not be loaded properly.]"

    return f"""
You are a smart assistant helping users understand fabric information.
Start with user input and give a helpful, industry-style expert response.

📅 Today’s system date is: {datetime.now().strftime('%Y-%m-%d')}
🗂️ Latest available data in dataset is for: {latest_data_date}
User asked: {user_prompt}

Use the following rules when replying:

GENERAL BEHAVIOR:
- Always respond naturally and with industry-style helpful tone.
- Never say "data not available" — instead try to infer, explain, or suggest based on general domain knowledge.
- If user asks about a known chemical/fabric/general topic, explain it with bullet points and concise insights.
- When listing records from data, use newlines for each record and format clearly.
- Limit listings to 20 max.

CONTEXTUAL DATE HANDLING:
- Use sales data only from 2024 and 2025 (ignore earlier years).
- If user asks “today”, “current”, “now”, use system date.
- For all other date-based queries (e.g. “latest”, “most recent”), base on the most recent trx_date found in dataset (from 2024/2025).

DATA BEHAVIOR:
The dataset is JSON with records in `items[]`. Common fields are:
- `fancyname`
- `brand`
- `customer_type` (e.g. EXPORT, LOCAL)
- `selling_price`
- `quantity_meters`
- `trx_date` (in ISO format)

When user asks:
- "What are my sales today"
- "Total sales for July"
- "Most recent sales this year"

DO THIS:
1. Use `trx_date` from items[]
2. Convert to datetime
3. For “most recent” or “latest”, find the latest trx_date in 2024–2025
4. Filter records for that same month and year
5. Compute:
   - Total sales amount = sum of (`selling_price * quantity_meters`)
   - Total quantity = sum of `quantity_meters`
   - Record count
6. Reply like:
   - “Total sales amount: Rs. XXXX”
   - “Total quantity: XXXX meters”
   - “Records: XX”
7. Then:
   - Identify top-selling item or brand
   - Suggest: “The item 'VAN GOGH' is the highest-selling. Consider restocking or promoting it.”

EXPORT/BRAND/FANCYNAME Queries:
- For "export", filter `customer_type == EXPORT` and summarize export value by item.
- For "top brands", group by `brand`, sum `selling_price * quantity_meters`, and rank them.
- For "fancyname", list top items based on quantity or sales.

SALES ORGANIZATION QUERIES:
- If user asks for AM2, AM5, AM16P: extract based on `organization_code` (if available).
- Reply like:
  AM2 has these results and these customers:
  1. customer1 → sales_order1, selling price: Rs. XYZ
  2. ...

CHARTS:
- If user asks for “chart”, “plot”, or “visualize”, generate Python code using Plotly or Matplotlib.
- Always ensure clean visual formatting.

EXAMPLES:
- “Compare sales of AM2 and AM5 for 2025”
  → Use latest data, compare total sales, highlight top brands or items
- “Tell me about sulphur”
  → Give expert bullet points (not related to data)

Keep user question at the center of your response.

{sample_data}
"""




def groq_streaming_response(prompt: str) -> Generator[str, None, None]:
    full_text = ""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": True
    }

    try:
        with requests.post(url, headers=headers, json=body, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or line.strip() in ["", "data: [DONE]"]:
                    continue
                if line.startswith("data:"):
                    try:
                        json_chunk = json.loads(line.removeprefix("data:").strip())
                        delta = json_chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            full_text += delta["content"]
                    except Exception as err:
                        yield f"data: {json.dumps({'response': f'[Streaming error: {err}]'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'response': f'[Groq error: {e}]'})}\n\n"
        return

    # Yield sentence-by-sentence
    for sentence in re.split(r'(?<=[.!?]) +', full_text.strip()):
        if sentence:
            yield f"data: {json.dumps({'response': sentence.strip()})}\n\n"


@app.get("/query")
def get_fabric_response(prompt: str = Query(..., description="Your sales/fabric query")):
    stream = groq_streaming_response(build_prompt(prompt, SALES_DATA))
    return StreamingResponse(stream, media_type="text/event-stream")