from paginator import paginate


def test_partial_last_page():
    assert paginate([1, 2, 3, 4, 5], 3, 2) == [5]


def test_past_the_end_is_empty():
    assert paginate([1, 2, 3], 5, 2) == []
