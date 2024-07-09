"""Contains a server lock class."""
# pylint: disable=too-few-public-methods

from src.operator.redis import RedisOperator


class ServerLockAdmin:
    """A lock class define a lock function."""

    SET_NAME = "server_lock"

    """A class for set lock."""
    def __init__(self, operator: RedisOperator):
        self.operator = operator

    def set_lock(self, value: str, ex: int = None, nx: bool = False) -> bool:
        """Set a lock"""
        return self.operator.set(self.SET_NAME, value=value, ex=ex, nx=nx)
