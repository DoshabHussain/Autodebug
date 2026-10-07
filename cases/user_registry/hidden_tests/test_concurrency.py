import threading

from registry.service import DuplicateUser, UserRegistry


def _hammer(reg, names):
    barrier = threading.Barrier(len(names))
    results = []

    def work(name):
        barrier.wait()
        try:
            results.append(("ok", reg.register(name, "x@example.com")["id"]))
        except DuplicateUser:
            results.append(("dup", None))
        except Exception as exc:  # noqa: BLE001
            results.append((type(exc).__name__, None))

    ts = [threading.Thread(target=work, args=(n,)) for n in names]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return results


def test_same_username_exactly_one_wins():
    for _ in range(5):
        res = _hammer(UserRegistry(), ["bob"] * 16)
        assert [r[0] for r in res].count("ok") == 1
        assert {r[0] for r in res} <= {"ok", "dup"}


def test_distinct_usernames_get_unique_ids():
    reg = UserRegistry()
    res = _hammer(reg, [f"user{i}" for i in range(16)])
    ids = [r[1] for r in res]
    assert reg.count() == 16 and len(set(ids)) == 16
