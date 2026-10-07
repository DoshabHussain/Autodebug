"""A tiny in-memory user registry that mimics a database-backed signup endpoint."""
import time


class DuplicateUser(Exception):
    """Username already taken (the API maps this to HTTP 409)."""


class IntegrityError(Exception):
    """Simulated database UNIQUE-constraint violation (the API would answer HTTP 500)."""


class UserRegistry:
    def __init__(self):
        self._rows = {}
        self._next_id = 1

    def _insert(self, username, email):
        if username in self._rows:
            raise IntegrityError(f"UNIQUE constraint failed: users.username ({username})")
        row = {"id": self._next_id, "username": username, "email": email}
        self._next_id += 1
        self._rows[username] = row
        return row

    def register(self, username, email):
        if username in self._rows:
            raise DuplicateUser(username)
        time.sleep(0.001)  # simulated round-trip to the database
        return self._insert(username, email)

    def get(self, username):
        return self._rows.get(username)

    def count(self):
        return len(self._rows)
