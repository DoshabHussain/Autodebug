POST /users intermittently returns HTTP 500 in production when two clients sign up with the same
username at (almost) the same time. Logs show "UNIQUE constraint failed: users.username".
A duplicate username should always produce a clean 409 (DuplicateUser), and exactly one of the
concurrent requests should succeed. It never happens when requests arrive one at a time.
