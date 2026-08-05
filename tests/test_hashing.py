from evalharness.hashing import canonical_json, config_hash, sha256_canonical, sha256_hex


def test_canonical_json_stable() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_sha256_hex() -> None:
    assert sha256_hex("hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_config_hash_changes_with_decode_params() -> None:
    base = dict(
        dataset_sha256="d",
        prompt_template_sha256="p",
        provider="ollama",
        model="m",
        resolved_version="v",
        harness_version="0.1.0",
    )
    h1 = config_hash(**base, decode_params={"temperature": 0.0})
    h2 = config_hash(**base, decode_params={"temperature": 0.1})
    assert h1 != h2


def test_sha256_canonical_dict() -> None:
    h = sha256_canonical({"x": 1})
    assert len(h) == 64
