import bob.pipelines as mario


def test_isinstance_nested():
    class A:
        pass

    class B:
        def __init__(self, o):
            self.o = o

    class C:
        def __init__(self, o):
            self.o = o

    o = C(B(A()))
    assert mario.utils.isinstance_nested(o, "o", C)
    assert mario.utils.isinstance_nested(o, "o", B)
    assert mario.utils.isinstance_nested(o, "o", A)

    o = C(B(object))
    assert mario.utils.isinstance_nested(o, "o", C)
    assert mario.utils.isinstance_nested(o, "o", B)
    assert not mario.utils.isinstance_nested(o, "o", A)
