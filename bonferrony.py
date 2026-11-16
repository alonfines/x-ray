"""Backward-compat shim: `bonferrony` now lives in conformal_methods/bonferrony.py.

Keeps `import bonferrony` (used by experiments 046-055) and `python bonferrony.py ...` working.
"""
import sys

from conformal_methods import bonferrony as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
