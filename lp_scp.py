"""Backward-compat shim: `lp_scp` now lives in conformal_methods/lp_scp.py.

Keeps `import lp_scp` (used by experiments 046-055) and `python lp_scp.py ...` working.
"""
import sys

from conformal_methods import lp_scp as _impl

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
