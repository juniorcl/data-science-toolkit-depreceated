from .statistical_tests.normal_tests   import apply_normal_tests
from .statistical_tests.kurtosis_test  import apply_kurtosis_test
from .statistical_tests.skewness_test  import apply_skewness_test
from .statistical_tests.dagostino_test import apply_dagostino_test

__all__ = [
    "apply_dagostino_test",
    "apply_normal_tests",
    "apply_kurtosis_test",
    "apply_skewness_test"
]