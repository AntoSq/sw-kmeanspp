"""
Self-contained path resolution: a single `data_root`, configurable via
`config/paths.yaml`, `--data-root`, or the `SWKMEANSPP_DATA_ROOT` env var --
no machine/account detection, no multi-directory cache search. Every path
lives under one root so a clean checkout works unmodified on any machine
(a reviewer's laptop, a GMD archival sandbox, a different HPC system).
"""

import os
import yaml


class PathResolver:
    def __init__(self, data_root=None, output_dir=None):
        self.data_root = os.path.abspath(data_root or 'data')
        self.output_dir = os.path.abspath(output_dir or 'output')
        self.raw_dir = os.path.join(self.data_root, 'raw')
        self.processed_dir = os.path.join(self.data_root, 'processed')
        self.cache_dir = os.path.join(self.data_root, 'cache')
        for d in (self.raw_dir, self.processed_dir, self.cache_dir, self.output_dir):
            os.makedirs(d, exist_ok=True)

    def raw(self, filename):
        return os.path.join(self.raw_dir, filename)

    def processed(self, filename):
        return os.path.join(self.processed_dir, filename)

    def cache(self, filename):
        return os.path.join(self.cache_dir, filename)

    def output(self, experiment_name, filename):
        exp_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        return os.path.join(exp_dir, filename)

    @classmethod
    def from_yaml(cls, path='config/paths.yaml', overrides=None):
        cfg = {}
        if os.path.exists(path):
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        data_root = (overrides or {}).get('data_root') or os.environ.get('SWKMEANSPP_DATA_ROOT') or cfg.get('data_root', './data')
        output_dir = (overrides or {}).get('output_dir') or cfg.get('output_dir', './output')
        return cls(data_root=data_root, output_dir=output_dir)
