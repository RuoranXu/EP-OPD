"""Training compatibility imports; metric implementations live under test/."""
from test.qa_metrics import (normalize_answer, token_f1, reference_text, annotation_f1, select_f1_checkpoint, score_prediction)
__all__ = ['normalize_answer', 'token_f1', 'reference_text', 'annotation_f1', 'select_f1_checkpoint', 'score_prediction']
