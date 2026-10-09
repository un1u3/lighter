# ## Task 0.2: Property Testing

@pytest.mark.task0_2
def test_sig():
    """Check properties of the sigmoid function, specifically
    that it is bounded, symmetric, and its two forms agree
    """
    assert sigmoid(0) == 0.5
    a = sigmoid(10000000)
    assert a > 0 and a < 1.0 + 1e-6
    b = sigmoid(-10000000)
    assert b > -1e-6 and b < 1
    c = sigmoid(5.0)
    d = sigmoid(-5.0)
    assert_close(1 - c, d)


@pytest.mark.task0_2
def test_transitive():
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    a, b, c = 1.0, 2.0, 3.0
    assert lt(a, b) == 1.0
    assert lt(b, c) == 1.0
    assert lt(a, c) == 1.0


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x: float, y: float, z: float) -> None:
    r"""
    Test the distributive property of multiplication over addition:
    z * (x + y) = z * x + z * y
    """
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_other(x: float, y: float) -> None:
    """
    Write a test that ensures some other property holds for your functions.

    Checks the commutativity of multiplication: x * y == y * x,
    and that negation is its own inverse: -(-x) == x.
    """
    assert_close(mul(x, y), mul(y, x))
    assert_close(neg(neg(x)), x)