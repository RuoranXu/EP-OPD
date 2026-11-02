from collections import Counter
import re
import string


def normalize_answer(text):
    text=''.join(c for c in str(text).lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def token_f1(prediction,reference):
    predicted=normalize_answer(prediction).split();gold=normalize_answer(reference).split()
    common=sum((Counter(predicted)&Counter(gold)).values())
    if not common:return 0.
    precision=common/len(predicted);recall=common/len(gold)
    return 2*precision*recall/(precision+recall)


def reference_text(annotation):
    if annotation.get('unanswerable'):return 'Unanswerable'
    if annotation.get('extractive_spans'):return ', '.join(annotation['extractive_spans'])
    if annotation.get('free_form_answer'):return annotation['free_form_answer']
    if annotation.get('yes_no') is not None:return 'Yes' if annotation['yes_no'] else 'No'
    raise ValueError('QASPER annotation contains no answer')


def annotation_f1(prediction,annotations):
    if not annotations:raise ValueError('QASPER references are empty')
    return max(token_f1(prediction,reference_text(a)) for a in annotations)




def score_prediction(prediction, annotations):
    """Maximum-reference token F1 and normalized exact match, never thresholded F1."""
    if not annotations:
        raise ValueError('QASPER references are empty')
    references=[reference_text(annotation) for annotation in annotations]
    return {'answer_f1':max(token_f1(prediction,ref) for ref in references),
            'exact_match':any(normalize_answer(prediction)==normalize_answer(ref) for ref in references)}
