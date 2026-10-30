"""Score complete JSONL predictions without changing prompts or selecting answers by gold.

python -m test.evaluate --split test --predictions outputs/test.jsonl
Optional --references supplies original QASPER JSONL id/gold_answers records.
For bundled QASPER use --qasper; external references use --qasper-references.
"""
import argparse
import hashlib
import json
from pathlib import Path

from test.paired_metrics import summarize, final_answer
from test.qa_metrics import score_prediction


def rows(path):
    with Path(path).open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def paired(pairs, predictions, references=None):
    if not pairs:
        raise ValueError('empty paired evaluation')
    keys = [(row['pair_id'], row['variant']) for row in predictions]
    expected = {(pair['pair_id'], variant) for pair in pairs for variant in pair['answers']}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('predictions must contain every expected pair/variant exactly once')
    signatures = {row.get('signature') for row in predictions}
    if len(signatures) != 1:
        raise ValueError('mixed model/protocol identities')
    report, details = summarize(pairs, dict(zip(keys, predictions)), references)
    report['capped_responses'] = sum(bool(row.get('hit_cap')) for row in predictions)
    return report, details


def qasper(references, predictions):
    lookup = {row['id']: row['gold_answers'] for row in references}
    ids = [row['id'] for row in predictions]
    if len(lookup) != len(references) or len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError('QASPER predictions/references must be complete, unique and matched')
    scores = [score_prediction(final_answer(row['text']), lookup[row['id']]) for row in predictions]
    if not scores:
        raise ValueError('empty QASPER evaluation')
    return {'questions': len(scores),
            'answer_f1': sum(score['answer_f1'] for score in scores) / len(scores),
            'exact_match': sum(score['exact_match'] for score in scores) / len(scores),
            'capped_responses': sum(bool(row.get('hit_cap')) for row in predictions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--cache')
    parser.add_argument('--split', choices=['development', 'test'], default='test')
    parser.add_argument('--references')
    parser.add_argument('--qasper-references')
    parser.add_argument('--qasper', action='store_true')
    args = parser.parse_args()
    path = Path(args.predictions)
    predictions = rows(path)
    from data.prepare import CACHE, references as bundled_references
    if args.qasper:
        args.qasper_references = str(bundled_references('development'))
    if args.qasper_references:
        result = qasper(rows(args.qasper_references), predictions)
    else:
        if not args.cache:
            args.cache = str(CACHE/'paired_cache.pt')
            if not args.references:
                args.references = str(bundled_references(args.split))
        if not Path(args.cache).is_file():
            parser.error('paired cache missing; run python -m data.prepare with local tokenizers first')
        import torch
        payload = torch.load(args.cache, map_location='cpu', weights_only=False)
        references = ({row['id']: row['gold_answers'] for row in rows(args.references)}
                      if args.references else None)
        result, details = paired(payload[args.split], predictions, references)
        result['original_references_sha256'] = (hashlib.sha256(Path(args.references).read_bytes()).hexdigest()
                                                if args.references else None)
        result['cache_sha256'] = hashlib.sha256(Path(args.cache).read_bytes()).hexdigest()
        identity_path = path.with_suffix('.identity.json')
        if identity_path.exists():
            result['identity'] = json.loads(identity_path.read_text(encoding='utf8'))
        path.with_suffix('.correctness.json').write_text(json.dumps(details), encoding='utf8')
    path.with_suffix('.summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
