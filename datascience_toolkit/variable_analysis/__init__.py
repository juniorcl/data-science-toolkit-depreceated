from .normal_tests   import apply_normal_tests
from .kurtosis_test  import apply_kurtosis_test
from .skewness_test  import apply_skewness_test
from .dagostino_test import apply_dagostino_test

__all__ = [
    "apply_dagostino_test",
    "apply_normal_tests",
    "apply_kurtosis_test",
    "apply_skewness_test"
]