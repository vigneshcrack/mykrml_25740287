from mykrml_25740287 import __version__


def test_package_exposes_version():
    assert __version__.count(".") == 2
