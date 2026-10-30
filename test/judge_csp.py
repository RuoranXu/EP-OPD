"""CSP v2: cover complete long responses in bounded spans, with lossless cache reuse."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import random
import time

from test.claims import (Judge, SEGMENT, SUPPORT, LABELS, digest, response_hash,
                                     read_rows, write_json, validate_coverage, validate_claims)
from test.evidence import ordered_blocks

LIMIT = 10000
CONTEXT = 512
CHUNK_SEGMENT = (SEGMENT + ' OUTPUT is one exact span of a longer response. CONTEXT_BEFORE and CONTEXT_AFTER '
    'are only for resolving sentence fragments and citations. Do not extract claims wholly within context. '
    'Assign a claim crossing a boundary to the OUTPUT span containing its final asserted words. '
    'Do not add claims from neighboring spans.')


def spans(text, maximum=LIMIT):
    """Contiguous spans; delimiters remain in the source and nothing is dropped."""
    if maximum < 1:
        raise ValueError('invalid span limit')
    result = []
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            for separator in ['\n\n', '\n', '. ', '; ', ' ']:
                boundary = text.rfind(separator, start + maximum // 2, end)
                if boundary >= 0:
                    end = boundary + len(separator)
                    break
        result.append((start, end))
        start = end
    return result


def reusable(row, label):
    return len(row['text']) <= LIMIT and label is not None and label['response_hash'] == response_hash(row)


def segment_response(text, question, call):
    memo = {}
    coverage = []
    def segment(start, end, short=False):
        data = {'QUESTION': question, 'OUTPUT': text[start:end]}
        system = SEGMENT if short else CHUNK_SEGMENT
        if not short:
            data.update(CONTEXT_BEFORE=text[max(0, start - CONTEXT):start], CONTEXT_AFTER=text[end:end + CONTEXT])
        key = digest([system, data])
        cached = key in memo
        try:
            if not cached:
                memo[key] = validate_claims(call(system, data, 4096))
            claims = copy.deepcopy(memo[key])
        except ValueError as exc:
            if 'truncated' not in str(exc) or end - start <= 256:
                raise
            middle = start + (end - start) // 2
            return segment(start, middle) + segment(middle, end)
        coverage.append(dict(start=start, end=end, source_sha256=hashlib.sha256(text[start:end].encode()).hexdigest(),
                             memoized=cached, extracted_claims=len(claims)))
        return claims
    if len(text) <= LIMIT:
        claims = segment(0, len(text), short=True)
    else:
        claims = []
        for start, end in spans(text):
            claims.extend(segment(start, end))
    ordered = sorted(coverage, key=lambda item: item['start'])
    if (not ordered or ordered[0]['start'] != 0 or ordered[-1]['end'] != len(text)
            or any(a['end'] != b['start'] for a, b in zip(ordered, ordered[1:]))):
        raise ValueError('CSP span coverage is not complete and contiguous')
    return claims, ordered


def main():
    import fcntl
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    from data.prepare import CACHE
    parser.add_argument('--cache', default=str(CACHE/'paired_cache.pt'))
    parser.add_argument('--split', required=True, choices=['development', 'test'])
    parser.add_argument('--legacy-directory', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    path = Path(args.predictions)
    lock = path.with_suffix('.complete_claim_worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read_rows(path)
    cache = Path(args.cache)
    pairs = torch.load(cache, map_location='cpu', weights_only=False)[args.split]
    validate_coverage(rows, pairs)
    lookup = {p['pair_id']: p for p in pairs}
    identity = dict(version='complete-csp-v2-chunked', split=args.split, judge='deepseek-flash',
        predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        prompts_sha256=digest([SEGMENT, CHUNK_SEGMENT, SUPPORT]), maximum_chunk_characters=LIMIT,
        context_characters=CONTEXT, coverage='all original characters; nonoverlapping ownership spans',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), blinded_to_method_and_gold=True)
    signature = digest(identity)
    manifest_path = path.with_suffix('.complete_claim_manifest.json')
    artifact = path.with_suffix('.complete_claim_labels.jsonl')
    progress_path = path.with_suffix('.complete_claim_progress.json')
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError('CSP v2 cache identity differs')
    elif artifact.exists():
        raise ValueError('unidentified CSP v2 labels')
    else:
        write_json(manifest_path, identity)
    completed = {}
    for label in read_rows(artifact) if artifact.exists() else []:
        if label['judge_signature'] != signature or label['response_hash'] in completed:
            raise ValueError('stale or duplicate CSP v2 labels')
        completed[label['response_hash']] = label
    expected = {response_hash(row) for row in rows}
    if not set(completed) <= expected:
        raise ValueError('extra CSP v2 labels')
    legacy = Path(args.legacy_directory)
    legacy_manifest_path = legacy / manifest_path.name
    legacy_labels_path = legacy / artifact.name
    prior_labels = {}
    prior_signature = None
    if legacy_manifest_path.exists() and legacy_labels_path.exists():
        prior = json.loads(legacy_manifest_path.read_text())
        for key in ['predictions_sha256', 'cache_sha256', 'split', 'judge', 'blinded_to_method_and_gold']:
            if prior[key] != identity[key]:
                raise ValueError('legacy CSP does not match current inputs: ' + key)
        if prior['prompts_sha256'] != digest([SEGMENT, SUPPORT]):
            raise ValueError('legacy CSP prompts differ')
        prior_signature = digest(prior)
        for label in read_rows(legacy_labels_path):
            if label['judge_signature'] != prior_signature or label['response_hash'] in prior_labels:
                raise ValueError('invalid legacy CSP cache')
            prior_labels[label['response_hash']] = label
    judge = Judge(os.environ['DEEPSEEK_API_KEY'])

    def label_response(row):
        old = prior_labels.get(response_hash(row))
        if reusable(row, old):
            item = copy.deepcopy(old)
            item.update(judge_signature=signature, legacy_judge_signature=prior_signature,
                        reused_identical_short_response=True, csp_protocol=identity['version'])
            return item
        pair = lookup[row['pair_id']]
        text = row['text']
        usage = {}
        calls = 0
        model = 'deepseek-flash'
        support_memo = {}

        def call(system, data, maximum):
            nonlocal calls, model
            value, used, model = judge.complete(system, data, maximum)
            calls += 1
            for k, v in used.items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            return value

        claims, ordered = segment_response(text, pair['question'], call)
        random.Random(response_hash(row)).shuffle(claims)
        blocks = ordered_blocks(pair['documents'][row['variant']])
        positions = {b['block_id']: i for i, b in enumerate(blocks)}
        decisions = []
        for claim in claims:
            if not claim['answer_relevant']:
                decisions.append(dict(claim, label='irrelevant'))
                continue
            ids = [v for v in claim['evidence_ids'] if v in positions]
            context = []
            for bid in ids:
                index = positions[bid]
                for i in range(max(0, index - 1), min(len(blocks), index + 2)):
                    if blocks[i] not in context:
                        context.append(blocks[i])
            if not context:
                decisions.append(dict(claim, label='absent', reason='no valid cited evidence'))
                continue
            data = dict(question=pair['question'], claim=claim['text'], cited_ids=ids, source=context)
            key = digest(data)
            if key not in support_memo:
                support_memo[key] = call(SUPPORT, data, 768)
            decision = support_memo[key]
            if decision.get('label') not in LABELS or not isinstance(decision.get('reason'), str):
                raise ValueError('invalid support decision')
            decisions.append(dict(claim, **decision))
        return dict(response_hash=response_hash(row), pair_id=row['pair_id'], variant=row['variant'],
                    claims=decisions, judge_model=model, judge_signature=signature,
                    csp_protocol=identity['version'], coverage=ordered, original_characters=len(text),
                    completed_judge_calls=calls, usage=usage, blinded_to_method_and_gold=True)

    todo = [r for r in rows if response_hash(r) not in completed]
    random.Random(42).shuffle(todo)
    failures = []
    access_failed = False
    with artifact.open('a', encoding='utf8', buffering=1) as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(todo)
        pending = {}
        def fill():
            while not access_failed and len(pending) < args.workers:
                row = next(iterator, None)
                if row is None:
                    break
                pending[pool.submit(label_response, row)] = row
        fill()
        while pending:
            done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in done:
                source = pending.pop(future)
                try:
                    item = future.result()
                except Exception as exc:
                    message = str(exc)
                    failures.append(dict(response_hash=response_hash(source), pair_id=source['pair_id'],
                                         variant=source['variant'], error=message))
                    access_failed = access_failed or 'API access unavailable' in message
                    continue
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
                completed[item['response_hash']] = item
                progress = dict(status='labeling', completed=len(completed), expected=len(rows),
                                csp_protocol=identity['version'], unix_time=time.time())
                write_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
            fill()
    if failures:
        write_json(path.with_suffix('.complete_claim_failures.json'), failures)
        raise RuntimeError(str(len(failures)) + ' CSP responses failed; successful labels preserved; ' + failures[0]['error'])
    if set(completed) != expected:
        raise ValueError('incomplete CSP v2 labels')
    by_pair = {p['pair_id']: [] for p in pairs}
    by_variant = {v: [] for v in ['Dorig', 'Dpara', 'Dirr', 'Ddrop', 'Dcf']}
    for row in rows:
        claims = [c for c in completed[response_hash(row)]['claims']
                  if c['answer_relevant'] and c['label'] != 'irrelevant']
        by_pair[row['pair_id']].extend(claims)
        if claims:
            by_variant[row['variant']].append(sum(c['label'] == 'supported' for c in claims) / len(claims))
    question_scores = {pid: sum(c['label'] == 'supported' for c in claims) / len(claims)
                       for pid, claims in by_pair.items() if claims}
    summary_path = path.with_suffix('.summary.json')
    summary = json.loads(summary_path.read_text())
    summary.update(claim_support_precision=sum(question_scores.values()) / len(question_scores) if question_scores else None,
        claim_evaluation_status='automated_blinded_labels_complete', csp_protocol=identity['version'],
        csp_question_coverage=len(question_scores), csp_total_questions=len(pairs),
        csp_labeled_responses=len(completed), csp_expected_responses=len(rows),
        csp_by_variant={v: {'claim_support_precision': sum(s) / len(s) if s else None,
                           'questions_with_relevant_claims': len(s)} for v, s in by_variant.items()},
        csp_aggregation='mean question-level precision across all five document variants',
        claim_judge='deepseek-flash', csp_judge_signature=signature,
        expert_label_agreement=None, expert_review_status='not_performed')
    for origin, item in summary.get('by_question_origin', {}).items():
        scores = [value for pid, value in question_scores.items()
                  if lookup[pid]['source'].get('question_origin', 'original_dataset_question') == origin]
        item.update(claim_support_precision=sum(scores) / len(scores) if scores else None,
                    claim_evaluation_status='automated_blinded_labels_complete', csp_protocol=identity['version'],
                    csp_question_coverage=len(scores))
    write_json(summary_path, summary)
    write_json(progress_path, dict(status='complete', completed=len(completed), expected=len(rows),
                                  csp_protocol=identity['version'], unix_time=time.time()))


if __name__ == '__main__':
    main()
