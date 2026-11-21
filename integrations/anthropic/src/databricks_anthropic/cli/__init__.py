"""The ``databricks-anthropic`` command: run the dispatcher on a laptop or VM.

``main.py`` is the entry point (``databricks_anthropic.cli.main:main``). ``config.py`` holds the saved settings and
keys, ``auth.py`` the Databricks login, ``lock.py`` the single-instance lock (POSIX only), and ``doctor.py`` the
prerequisite checks.
"""
