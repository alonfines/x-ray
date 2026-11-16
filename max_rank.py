"""Backward-compat shim: `max_rank` now lives in conformal_methods/max_rank.py.

Keeps `import max_rank` (used by experiments 046-055) and `python max_rank.py ...` working.
"""
import sys

from conformal_methods import max_rank as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
