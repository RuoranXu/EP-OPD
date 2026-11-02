import numpy as np
METHODS=["base","ep_opd"]

def ratios(samples, count=10000, seed=42):
    """samples: method -> (unit ID, paper ID, {metric: (numerator, denominator)})."""
    ids = [{r[0] for r in samples[m]} for m in METHODS]
    if ids[0] != ids[1] or any((len(samples[m]) != len(ids[i]) for (i, m) in enumerate(METHODS))):
        raise ValueError('incomplete or duplicate paired metric units')
    paper_lookup = [{r[0]: r[1] for r in samples[m]} for m in METHODS]
    if paper_lookup[0] != paper_lookup[1]:
        raise ValueError('paired source-paper assignments differ')
    papers = sorted(set(paper_lookup[0].values()))
    columns = sorted(set().union(*(set(r[2]) for m in METHODS for r in samples[m])))
    paper_index = {p: i for (i, p) in enumerate(papers)}
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(papers), np.full(len(papers), 1 / len(papers)), size=count)
    reports = {}
    for metric in columns:
        reports[metric] = {}
        values = {}
        for method in METHODS:
            totals = np.zeros((len(papers), 2), dtype=np.float64)
            for (_, paper, payload) in samples[method]:
                totals[paper_index[paper]] += payload.get(metric, (0.0, 0.0))
            point = totals.sum(0)
            aggregate = draws @ totals
            sample = np.divide(aggregate[:, 0], aggregate[:, 1], out=np.full(count, np.nan), where=aggregate[:, 1] > 0)
            values[method] = sample
            valid = sample[np.isfinite(sample)]
            reports[metric][method] = dict(estimate=float(point[0] / point[1]) if point[1] > 0 else None, ci95=np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None, numerator=float(point[0]), denominator=float(point[1]), valid_bootstrap_replicates=int(len(valid)))
        delta = values['ep_opd'] - values['base']
        valid = delta[np.isfinite(delta)]
        (left, right) = (reports[metric]['base']['estimate'], reports[metric]['ep_opd']['estimate'])
        reports[metric]['delta'] = dict(estimate=right - left if left is not None and right is not None else None, ci95=np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None, valid_bootstrap_replicates=int(len(valid)))
    return dict(metrics=reports, units=len(ids[0]), papers=len(papers), bootstrap_replicates=count, cluster='source_paper', random_seed=seed, paired_draws_shared_across_methods=True)
