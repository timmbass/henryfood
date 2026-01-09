import importlib


def test_imports():
    modules = [
        "src.ingest.usda_client",
        "src.ingest.off_ingest",
        "src.ingest.liljebo",
        "src.ingest.sighi",
    ]
    for m in modules:
        importlib.import_module(m)
    assert True


def test_placeholder_ingest():
    # Placeholder: add tests for ingest functions
    assert True
