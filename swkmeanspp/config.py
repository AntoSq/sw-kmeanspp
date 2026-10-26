"""YAML config loading and validation for domains and experiments."""

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_domains(path='config/domains.yaml'):
    return load_yaml(path)


def _weights_for_run(experiment, run):
    """Resolve {variable: w_t} and w_s for one run, merging experiment- and run-level overrides."""
    variables = experiment['variables']
    w_s = run.get('w_s', experiment.get('w_s'))
    if 'w_t' in run:
        w_t = dict(run['w_t'])
    elif 'w_t' in experiment:
        w_t = dict(experiment['w_t'])
    elif len(variables) == 1:
        w_t = {variables[0]: 1 - w_s}
    else:
        # Equal per-variable temporal weight, splitting what's left after w_s
        # (manuscript Sect. 2.7: "by e.g. providing same value to all w_t,v").
        share = (1 - w_s) / len(variables)
        w_t = {v: share for v in variables}
    return w_t, w_s


def load_experiment(path):
    """
    Load and validate an experiment config. Returns the raw dict with each
    `runs` entry augmented with resolved `w_t`/`w_s`, after checking:
    - every run's w_t + w_s sums to 1 (within floating-point tolerance);
    - `experimental.enable_climatology_term` is explicit opt-in (default False) --
      none of the 5 published experiment configs in this repo set it.
    """
    cfg = load_yaml(path)
    experimental = cfg.get('experimental', {})
    enable_climatology = experimental.get('enable_climatology_term', False)

    for run in cfg['runs']:
        w_t, w_s = _weights_for_run(cfg, run)
        total = w_s + sum(w_t.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Run {run.get('label')!r} in {path}: w_t + w_s = {total}, must sum to 1")
        run['w_t'], run['w_s'] = w_t, w_s
        run['beta'] = experimental.get('beta', 0.0) if enable_climatology else 0.0

    cfg['enable_climatology_term'] = enable_climatology
    return cfg
