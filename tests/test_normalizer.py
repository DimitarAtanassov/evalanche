from evalharness.scoring.normalizer import Normalizer, NormalizerConfig


def test_normalizer_strips_articles_and_punctuation() -> None:
    n = Normalizer(NormalizerConfig())
    assert n.normalize("The Answer: 42!") == "answer 42"


def test_normalizer_config_id_stable() -> None:
    n = Normalizer(NormalizerConfig())
    assert len(n.config_id) == 64
