import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class CriterionScore:
    """Score for a specific evaluation criterion"""
    score: int
    explanation: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CriterionScore':
        return cls(
            score=data.get('score', 0),
            explanation=data.get('explanation', '')
        )

@dataclass
class RequirementEvaluation:
    """Evaluation results for a requirement"""
    criterion_scores: Dict[str, CriterionScore] = field(default_factory=dict)
    overall_score: float = 0.0
    overall_status: str = "Not Evaluated"
    suggested_rewrite: str = ""
    error: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RequirementEvaluation':
        """Create an evaluation object from JSON string"""
        try:
            data = json.loads(json_str)
            eval_obj = cls()

            # Validate and parse criterion_scores
            if 'criterion_scores' in data:
                if isinstance(data['criterion_scores'], dict):
                    for criterion, score_data in data['criterion_scores'].items():
                        eval_obj.criterion_scores[criterion] = CriterionScore.from_dict(score_data)
                else:
                    # Attempt to cast to a dictionary if possible
                    try:
                        data['criterion_scores'] = dict(data['criterion_scores'])
                        for criterion, score_data in data['criterion_scores'].items():
                            eval_obj.criterion_scores[criterion] = CriterionScore.from_dict(score_data)
                    except (TypeError, ValueError):
                        raise ValueError("Invalid format for 'criterion_scores'. Could not convert to a dictionary.")

            eval_obj.overall_score = data.get('overall_score', 0.0)
            eval_obj.overall_status = data.get('overall_status', "Error")
            eval_obj.suggested_rewrite = data.get('suggested_rewrite', "")
            eval_obj.error = data.get('error')

            return eval_obj

        except Exception as e:
            # Return an error evaluation
            error_eval = cls()
            error_eval.error = f"Failed to parse evaluation JSON: {str(e)}"
            error_eval.overall_status = "Error"
            return error_eval
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'criterion_scores': {
                criterion: {
                    'score': score.score,
                    'explanation': score.explanation
                } for criterion, score in self.criterion_scores.items()
            },
            'overall_score': self.overall_score,
            'overall_status': self.overall_status,
            'suggested_rewrite': self.suggested_rewrite,
            'error': self.error
        }

@dataclass
class Requirement:
    """Represents a single requirement with its properties"""
    id: str
    text: str
    source_file: str
    evaluation: Optional[RequirementEvaluation] = None
    additional_attributes: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], source_file: str) -> 'Requirement':
        """Create a requirement from a dictionary"""
        # Try to find ID and text fields (handle different possible field names)
        req_id = None
        req_text = None
        
        id_fields = ['ID', 'Id', 'id', 'Identifier', 'identifier']
        text_fields = ['Text', 'text', 'Requirement', 'requirement', 'Description', 'description']
        
        for field in id_fields:
            if field in data:
                # Convert to string and handle numpy arrays
                value = data[field]
                if hasattr(value, 'item'):  # Check if it's a numpy scalar
                    value = value.item()
                req_id = str(value)
                break
                
        for field in text_fields:
            if field in data:
                # Convert to string and handle numpy arrays
                value = data[field]
                if hasattr(value, 'item'):  # Check if it's a numpy scalar
                    value = value.item()
                req_text = str(value)
                break
        
        if not req_id or not req_text:
            raise ValueError("Dictionary does not contain required ID and text fields")
        
        # Get additional attributes (excluding ID and text)
        additional = {k: v for k, v in data.items() 
                     if k not in id_fields and k not in text_fields}
        
        return cls(
            id=req_id,
            text=req_text,
            source_file=source_file,
            additional_attributes=additional
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            'ID': self.id,
            'Text': self.text,
            'Source': self.source_file
        }
        
        # Add evaluation data if available
        if self.evaluation:
            result['Evaluation'] = self.evaluation.to_dict()
            
        # Add any additional attributes
        result.update(self.additional_attributes)
        
        return result
