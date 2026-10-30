"""Keep both teacher and student replay inside their shared context budget."""


def response_cap(student_length, teacher_length, requested_cap, context_window):
    if min(student_length, teacher_length, requested_cap, context_window) < 1:
        raise ValueError('length budgets must be positive')
    available = context_window - max(student_length, teacher_length)
    if available < 1:
        raise ValueError('prompt leaves no response space for teacher/student replay')
    return min(requested_cap, available)


def pair_caps(pair, requested_cap, context_window):
    return [response_cap(len(pair['student_prompts'][v]), len(pair['teacher_prompts'][v]),
                         requested_cap, context_window) for v in ['Dorig', 'Dcf']]


def batch_cap(pairs, requested_cap, context_window):
    # Replay pads responses to a common width. Reserve room for the longest
    # teacher OR student prefix in the entire microbatch, not just each row.
    return min(c for pair in pairs for c in pair_caps(pair, requested_cap, context_window))


def check_response_health(history, config):
    """Stop on consecutive broadly truncated updates; never skip their evidence."""
    if not config:
        return None
    count = int(config['consecutive_updates'])
    threshold = float(config['maximum_cap_fraction'])
    minimum = int(config.get('minimum_update', count))
    if count < 1 or not 0 <= threshold <= 1:
        raise ValueError('invalid response health gate')
    if len(history) < count or history[-1]['step'] < minimum:
        return None
    tail = history[-count:]
    if all(row['documents'] > 0 and row['capped'] / row['documents'] > threshold for row in tail):
        return {'reason': 'consecutive excessive response truncation; diagnose before continuing',
                'threshold': threshold, 'updates': [row['step'] for row in tail],
                'cap_fractions': [row['capped']/row['documents'] for row in tail]}
    return None
