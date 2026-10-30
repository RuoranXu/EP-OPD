"""One shared completion boundary: preserve the SFT END marker before EOS."""
from ._termination import (EOS_IDS,STUDENT_EOS,END_MARKER,ID_PATTERN,SYSTEM,content,encode_prefix,
    inspect_completion,repeated,semantic_stop_kl,stop_nll,teacher_stop_positions)

VERSION='complete-marker-stop-v2-20261028'

__all__ = ['EOS_IDS','STUDENT_EOS','END_MARKER','ID_PATTERN','SYSTEM','VERSION',
           'content','encode_prefix','inspect_completion','repeated','semantic_stop_kl',
           'stop_nll','teacher_stop_positions','token_repetition','boundary_reason',
           'CompletionBoundaryProcessor','generation_options','make_replay']


def token_repetition(tokens):
    tail=list(tokens[-512:])
    if len(tail)<128:return False
    grams=[tuple(tail[i:i+4]) for i in range(len(tail)-3)]
    return 1-len(set(grams))/len(grams)>.60


def boundary_reason(text):
    if END_MARKER in text:return 'completion_marker'
    if repeated(text[-16000:]):return 'repetition_abort'
    return None


_TOKENIZERS={}


class CompletionBoundaryProcessor:
    """A complete evidence line still needs its learned END marker, not premature EOS.

    Inspect a bounded token suffix. Full-response decoding and repeated line
    scans on every newline made pathological 32K baseline outputs expensive.
    """
    def __init__(self,tokenizer_path,interval=32):self.path=str(tokenizer_path);self.interval=interval

    def __call__(self,prompt_ids,response_ids,scores):
        if not response_ids:return scores
        if len(response_ids)%self.interval==0 and token_repetition(response_ids):
            scores.fill_(float('-inf'));scores[STUDENT_EOS]=0.;return scores
        if self.path not in _TOKENIZERS:
            from transformers import AutoTokenizer
            _TOKENIZERS[self.path]=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        tokenizer=_TOKENIZERS[self.path]
        last=tokenizer.decode(response_ids[-1:],skip_special_tokens=True)
        if '>' not in last and '\n' not in last and len(response_ids)%self.interval:return scores
        # Any END marker would have stopped at its first occurrence; a bounded
        # suffix is sufficient to recognize the newly completed marker.
        text=tokenizer.decode(response_ids[-1024:],skip_special_tokens=True)
        if boundary_reason(text):scores.fill_(float('-inf'));scores[STUDENT_EOS]=0.
        return scores


def generation_options(tokenizer_path,guard=True):
    options=dict(stop_token_ids=list(EOS_IDS),detokenize=True,skip_special_tokens=False)
    if guard:options.update(stop=[END_MARKER],include_stop_str_in_output=True,
        logits_processors=[CompletionBoundaryProcessor(tokenizer_path)])
    return options


def make_replay(tokenizer,tokens,anchor_ids,allowed_ids):
    values=list(tokens);text=tokenizer.decode(values,skip_special_tokens=True)
    checked=inspect_completion(text,allowed_ids,require_marker=True)
    def fallback(reason):
        return list(anchor_ids),False,dict(reason=reason,teacher_saw_bad_prefix=False,
            discarded_tokens=len(values),fallback_to_supervision=True,stop_protocol=VERSION)
    # Native EOS without the learned marker is not a license to move the clean
    # EOS label to an earlier, different prefix.
    if END_MARKER not in text:return fallback('missing_complete_end_marker')
    if not checked.valid:return fallback(checked.reason)
    rendered=tokenizer.decode(values,skip_special_tokens=False);end=rendered.find(END_MARKER)+len(END_MARKER)
    prefix=[]
    for i,token in enumerate(values):
        if token in EOS_IDS:break
        decoded=tokenizer.decode(values[:i+1],skip_special_tokens=False)
        if len(decoded)>end and decoded[end:].strip():return fallback('token_crosses_completion_boundary')
        prefix.append(token)
        if len(decoded)>=end:break
    if not prefix or END_MARKER not in tokenizer.decode(prefix,skip_special_tokens=True):
        return fallback('incomplete_marker_token_boundary')
    if token_repetition(prefix):return fallback('repetitive_tokens_before_completion')
    clean=prefix+[STUDENT_EOS]
    return clean,True,dict(reason='complete_marker_preserved',teacher_saw_bad_prefix=False,
        native_eos_token_present=bool(values and values[-1] in EOS_IDS),
        natural_eos_unmeasured_in_guarded_rollouts=True,sampled_prefix_preserved=True,
        end_marker_preserved=True,discarded_tokens=max(0,len(values)-len(clean)),
        discarded_suffix=checked.discarded_suffix or len(values)>len(clean),fallback_to_supervision=False,stop_protocol=VERSION)
