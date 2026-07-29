import importlib

def test_exports_present():
    m = importlib.import_module('adapter')
    assert hasattr(m, 'infection_spread_model') and callable(getattr(m, 'infection_spread_model'))
