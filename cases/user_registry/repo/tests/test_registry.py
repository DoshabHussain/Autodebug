import pytest

from registry.service import DuplicateUser, UserRegistry


def test_register_and_get():
    reg = UserRegistry()
    row = reg.register("alice", "a@example.com")
    assert row["id"] == 1
    assert reg.get("alice")["email"] == "a@example.com"


def test_sequential_duplicate_is_rejected():
    reg = UserRegistry()
    reg.register("alice", "a@example.com")
    with pytest.raises(DuplicateUser):
        reg.register("alice", "other@example.com")


def test_count():
    reg = UserRegistry()
    reg.register("a", "a@x.com")
    reg.register("b", "b@x.com")
    assert reg.count() == 2
