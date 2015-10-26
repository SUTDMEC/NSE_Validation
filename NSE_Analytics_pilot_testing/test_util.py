"""
Unit tests for util.py
"""

from util import great_circle_dist
from nose.tools import assert_almost_equal



def test_great_circle_dist():
    """
    test great circle distance funtion
    """
    point1 = (1.3, 103.0)
    # distance from a point to itself is zero
    assert_almost_equal(0.0, great_circle_dist(point1, point1))
    # distance from CREATE to SUTD (rounded to 100ms)
    p_create = (1.303826, 103.773890)
    p_sutd = (1.341221, 103.963234)
    assert_almost_equal(21.5, round(great_circle_dist(p_create, p_sutd), ndigits=1))
    assert_almost_equal(21.5, round(great_circle_dist(p_sutd, p_create), ndigits=1))
