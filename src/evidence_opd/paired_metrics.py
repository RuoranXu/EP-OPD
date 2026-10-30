"""Training compatibility imports; metric implementations live under test/."""
from test.paired_metrics import (SCORER_VERSION, final_answer, normalize, numeric, correct, summarize, aggregate_rows)
__all__ = ['SCORER_VERSION', 'final_answer', 'normalize', 'numeric', 'correct', 'summarize', 'aggregate_rows']
