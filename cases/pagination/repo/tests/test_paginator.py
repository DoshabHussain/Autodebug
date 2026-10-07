from paginator import page_count, paginate

DATA = [1, 2, 3, 4, 5]


def test_first_page():
    assert paginate(DATA, 1, 2) == [1, 2]


def test_second_page():
    assert paginate(DATA, 2, 2) == [3, 4]


def test_page_count():
    assert page_count(5, 2) == 3
