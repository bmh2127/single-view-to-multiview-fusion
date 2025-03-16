from pipelines.common import PACKAGES, packages


def test_packages_returns_package_version():
    result = packages("torch")
    assert result["torch"] == PACKAGES["torch"]


def test_packages_returns_package_without_version():
    result = packages("random-package")
    assert result["random-package"] == ""


def test_packages_returns_multiple_packages():
    result = packages("torch", "numpy")
    assert "torch" in result
    assert "numpy" in result
