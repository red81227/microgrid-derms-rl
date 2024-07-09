"""Contains a input class."""
# pylint: disable=too-few-public-methods

from src.operator.redis import RedisOperator


class InputAdmin:
    """A input class define a input function."""

    SET_NAME = "system_client_input"

    """A class for set input."""
    def __init__(self, operator: RedisOperator):
        self.operator = operator
        self.operator.expire(self.SET_NAME, 3)

    def hmset_input(self, mapping: dict) -> bool:
        """Mset a input"""
        return self.operator.hmset(name=self.SET_NAME, mapping=mapping)
    
    def hmgetall_input(self):
        """Mget the values of all the given keys."""
        return self.operator.hgetall(name=self.SET_NAME)