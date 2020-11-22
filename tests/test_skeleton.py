# -*- coding: utf-8 -*-

import pytest
from pydabax.skeleton import fib

__author__ = "Julian Mars"
__copyright__ = "Julian Mars"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
