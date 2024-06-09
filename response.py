from mcqgenerator import generate_evaluate_chain
from streamlit import RESPONSE_JSON, mcq_count, subject, tone


import json


response=generate_evaluate_chain({
    "text": text,
    "number": mcq_count,
    "subject": subject,
    "tone": tone,
    "response_json": json.dumps(RESPONSE_JSON)
})