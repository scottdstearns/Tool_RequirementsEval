from openai import OpenAI
import os, json

BASE  = os.getenv("OPENAI_BASE_URL","http://litellm:4000/v1")
KEY   = os.getenv("OPENAI_API_KEY","sk-1234")
DEFAULT_MODEL = os.getenv("CHAT_MODEL","azure-gpt-5")  # must exist in litellm-config.yaml

class OpenWebUIClient:
    def __init__(self, model: str | None = None):
        self.client = OpenAI(base_url=BASE, api_key=KEY)
        self.model = (model or DEFAULT_MODEL or "azure-gpt-4o").strip()
        if not self.model:
            # absolute last-resort guard
            self.model = "azure-gpt-4o"
        print(f"[LLM] Using model alias: {self.model}")

    def evaluate_requirement(self, requirement_text: str, evaluation_criteria: str, context: str | None = None) -> str:
        system_prompt = (
            "You are an expert requirements engineer (INCOSE). "
            "Return ONLY JSON with fields: "
            "criterion_scores{criterion:{score,explanation}}, overall_score, overall_status, suggested_rewrite. "
            "overall_score is the average of criterion scores (0-1) rounded to 2 decimals. "
            "overall_status = 'Pass'(>=0.80), 'Needs-Revision'(0.60–0.79), else 'Fail'."
        )
        user = f'Requirement: "{requirement_text}"\nCriteria:\n{evaluation_criteria}'
        if context:
            user += f"\n\nContext (from KB):\n{context}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,                      # <- ALWAYS set
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user}],
                response_format={"type":"json_object"},
                timeout=90
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[LLM] Chat error on '{self.model}': {e}")
            return json.dumps({
                "criterion_scores": {},
                "overall_score": 0.0,
                "overall_status": "Fail",
                "suggested_rewrite": "Evaluation service unavailable; please retry."
            })

