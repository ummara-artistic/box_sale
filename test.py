import streamlit as st
import json
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load .env API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or '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'

# Load JSON data
@st.cache_data
def load_sales_data():
    with open(r"D:/box_sale/sales_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Define streaming LLM function
def query_llm_stream(prompt, data=None):
    system_prompt = f"""
You are a smart assistant helping users understand fabric information.
Start with user input and give a helpful, industry-style expert response.
If ask related to listing or any kind of query related to data in json, chng line if u r going to show records also 
if ask like composition of sulphur make it answer like general query, like Sulphur is this and that, chnge line for bullet points
for advantage for anything except data, make it start with the user input not like, this not include in our data bluh bluh
If user asks something like "what are my sales today", calculate the total sales amount, total quantity, and total records for today (based on trx_date). 

if ask like :
Compare sales figures of AM2 and AM5 for the year 2025,
u have to check the most recent sales and then give query 
which has 11.50 oz
Sample answer: ERNAN brand has 11.50 oz and its has the selling_price of 794.31 with quantity 4434 in meters make it like that

Check organization code, and take it if ask about AM2 or something, it should give answer based on its results like 
ANs: Am2 has these results and has these customer like
1. artistimiliner that has this slaes order and has this sleing poin
2. 
GENERAL BEHAVIOR:
- Always respond naturally and with industry-style helpful tone.
- Never say "data not available" — instead try to infer, explain, or suggest based on general domain knowledge.
- If user asks about a known chemical/fabric/general topic, explain it with bullet points and concise insights.
- When listing records from data, use newlines for each record and format clearly.

if ask about like team
make it align like 
1. 
2. 
not like numbering above, and answer below
DATA BEHAVIOR:
The dataset is JSON with records in `items[]`. Common fields are:
- `fancyname`
- `brand`
- `customer_type` (e.g. EXPORT, LOCAL)
- `selling_price`
- `quantity_meters`
- `trx_date` (in ISO format) = date

If user asks for:
- "What are my sales today"
- "Total sales for July"
- "Most recent sales this year"
Do this:
1. Use `trx_date` from `items[]`
2. Convert to datetime
3. Find latest trx_date (ignore today’s system date)
4. Filter all records for that same **month and year**
5. Compute:
   - Total sales amount = sum of (`selling_price * quantity_meters`)
   - Total quantity = sum of `quantity_meters`
   - Count of records
6. Respond like:
   - “Most recent sales are for July 2025”
   - “Total sales amount: Rs. XXXX”
   - “Total quantity: XXXX meters”
   - “Records: XX”
7. Then:
   - Find top-selling item (by total value)
   - Suggest: “The item 'VAN GOGH' is the highest-selling. Consider restocking or promoting it.”
   - OR highlight best-performing brand.

EXPORT/BRAND/FANCYNAME Queries:
- For "export", filter `customer_type == EXPORT` and summarize export value by item.
- For "top brands", group by `brand`, sum `selling_price * quantity_meters`, and rank them.
- For "fancyname", list top items based on quantity or sales.

CHARTS:
If user asks:
- "Plot", "chart", "graph", or "visualize":
Return a clean Python code block using Plotly or Matplotlib, showing data from `items[]`.

show 20 listing for each
REMEMBER:
Should give 20 listing
Show tables as newline-formatted lists

Keep user question at the center of your response

Don’t explain what you can’t do — always try to provide something useful

📅 Today’s date is {datetime.now().strftime('%Y-%m-%d')}.
User asked: {prompt}
{f"Here is some sample data:\n{json.dumps(data, indent=2)[:8000]}" if data else ""}
"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": system_prompt}],
        "temperature": 0.7,
        "stream": True
    }

    with requests.post(url, headers=headers, json=body, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or line.strip() in ["", "data: [DONE]"]:
                continue
            if line.startswith("data:"):
                try:
                    chunk = json.loads(line.removeprefix("data:").strip())
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                except Exception as err:
                    yield f"\n[Error: {err}]"

# === Streamlit UI ===
st.title("📊 Fabric Sales Smart Assistant (Groq + LLaMA3)")

query = st.text_input("Ask about your sales data or fabric info:", placeholder="e.g. What are my sales today?")

if query:
    sales_data = load_sales_data()
    st.write("🤖 Assistant response:")
    response_area = st.empty()
    full_response = ""
    for chunk in query_llm_stream(query, sales_data):
        full_response += chunk
        response_area.markdown(full_response)
