from authentication.passwords import hash_password, needs_rehash, verify_password


def test_hash_is_salted_and_verifies():
    password = "S3cure-Pass!"
    hashed = hash_password(password)
    assert hashed != password
    assert hashed.startswith("$2")
    assert verify_password(password, hashed)
    assert not verify_password("wrong-password", hashed)


def test_same_password_produces_different_hashes():
    assert hash_password("samepassword") != hash_password("samepassword")


def test_empty_password_rejected():
    try:
        hash_password("")
    except ValueError:
        return
    raise AssertionError("Empty password should raise ValueError")


def test_needs_rehash_for_plaintext():
    assert needs_rehash(None) is True
    assert needs_rehash("not-a-bcrypt-hash") is True
    assert needs_rehash(hash_password("another-pass")) is False
