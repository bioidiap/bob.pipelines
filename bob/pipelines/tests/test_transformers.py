import bob.pipelines as mario


def test_str_to_types():
    samples = [
        mario.Sample(None, id="1", flag="True"),
        mario.Sample(None, id="2", flag="False"),
    ]
    transformer = mario.transformers.Str_To_Types(
        fieldtypes=dict(id=int, flag=mario.transformers.str_to_bool)
    )
    transformer.transform(samples)
    assert samples[0].id == 1
    assert samples[0].flag is True
    assert samples[1].id == 2
    assert samples[1].flag is False
