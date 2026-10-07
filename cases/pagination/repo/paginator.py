def paginate(items, page, size):
    """Return the items on a 1-indexed page."""
    start = page * size
    return list(items[start:start + size])


def page_count(total, size):
    return -(-total // size)
