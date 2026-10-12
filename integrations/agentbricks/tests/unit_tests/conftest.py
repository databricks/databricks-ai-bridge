"""Shared fixtures for the Agent Bricks CLI unit tests.

Service tests inject their collaborators directly; a directory-wide patch of the old deploy
module would mask the new boundaries and fails because that function no longer exists.
"""
