# app/utils/requirements_evaluator.py
import json
from typing import Callable, Iterable, List, Optional, Dict, Any, Tuple
from app.utils.openwebui_client import OpenWebUIClient as AzureOpenAIClient
from app.models.requirement import Requirement, RequirementEvaluation

class RequirementsEvaluator:
    def __init__(self, criteria: str, retriever: Optional[Callable[[str], str]] = None, model: Optional[str] = None):
        self.criteria = criteria
        self.retriever = retriever          # <-- may be None if "Use KB" is off
        self.client = AzureOpenAIClient(model=model)  # uses OPENAI_BASE_URL / KEY / CHAT_MODEL envs

    def _normalize(self, ev: Dict[str, Any]) -> RequirementEvaluation:
        """
        Accepts raw dict from the model and returns a normalized RequirementEvaluation.
        No from_json() needed; we construct the dataclass ourselves.
        """
        scores: Dict[str, Dict[str, Any]] = {}
        for crit, d in (ev.get("criterion_scores") or {}).items():
            s = float(d.get("score", 0) or 0)
            s = max(0.0, min(1.0, s))
            scores[crit] = {"score": round(s, 2), "explanation": str(d.get("explanation", ""))}

        overall = 0.0
        if scores:
            overall = sum(v["score"] for v in scores.values()) / len(scores)
        overall = round(overall, 2)

        status = "Pass" if overall >= 0.80 else ("Needs-Revision" if overall >= 0.60 else "Fail")
        return RequirementEvaluation(
            criterion_scores=scores,
            overall_score=overall,
            overall_status=status,
            suggested_rewrite=str(ev.get("suggested_rewrite", "")),
        )

    '''def evaluate_requirements(self, requirements: Iterable[Requirement]) -> List[RequirementEvaluation]:
        results: List[RequirementEvaluation] = []
        for req in requirements:
            # 1) optional KB context
            ctx: Optional[str] = self.retriever(req.text) if self.retriever else None

            # 2) call the model (returns JSON string)
            raw = self.client.evaluate_requirement(req.text, self.criteria, context=ctx)

            # 3) parse -> normalize -> collect
            try:
                ev_dict = json.loads(raw)
            except Exception:
                # be defensive: if the model slipped some prose, try to strip to last JSON block
                start = raw.find("{")
                end = raw.rfind("}")
                ev_dict = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {}
            results.append(self._normalize(ev_dict))
        return results
    '''
    def evaluate_requirements(self, requirements: Iterable[Requirement]) -> List[Tuple[Requirement, RequirementEvaluation]]:
        pairs: List[Tuple[Requirement, RequirementEvaluation]] = []
        for req in requirements:
            ctx = self.retriever(req.text) if self.retriever else None
            raw = self.client.evaluate_requirement(req.text, self.criteria, context=ctx)
            try:
                ev_dict = json.loads(raw)
            except Exception:
                start, end = raw.find("{"), raw.rfind("}")
                ev_dict = json.loads(raw[start:end+1]) if start != -1 and end != -1 else {}
            ev = self._normalize(ev_dict)
            pairs.append((req, ev))
        return pairs

    '''def generate_evaluation_report(self, evaluations: List[RequirementEvaluation]):
        import pandas as pd
        rows = []
        for ev in evaluations:
            row = {
                "Overall Score": ev.overall_score,
                "Status": ev.overall_status,
                "Suggested Rewrite": ev.suggested_rewrite,
            }
            # flatten per-criterion scores
            for crit, d in ev.criterion_scores.items():
                row[f"{crit} (score)"] = d.get("score", 0.0)
                row[f"{crit} (explanation)"] = d.get("explanation", "")
            rows.append(row)
        return pd.DataFrame(rows)
    '''
    def generate_evaluation_report(self, pairs: List[Tuple[Requirement, RequirementEvaluation]]):
        import pandas as pd
        rows = []
        for req, ev in pairs:
            row = {
                "ID": req.id,
                "Requirement Text": req.text,
                "Overall Score": ev.overall_score,
                "Status": ev.overall_status,
                "Suggested Rewrite": ev.suggested_rewrite,
            }
            for crit, d in ev.criterion_scores.items():
                row[f"{crit} (score)"] = d.get("score", 0.0)
                row[f"{crit} (explanation)"] = d.get("explanation", "")
            rows.append(row)
        # columns A/B are now ID and Text
        return pd.DataFrame(rows)
