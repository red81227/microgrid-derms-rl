"""Contains a output class."""
# pylint: disable=too-few-public-methods

from src.operator.redis import RedisOperator


class OutputPAdmin:
    """A lock class define a output function."""

    SET_NAME = "output_p"

    """A class for set output p."""
    def __init__(self, operator: RedisOperator):
        self.operator = operator

    def set_output_p(self, value: str, ex: int = None, nx: bool = False) -> bool:
        """Set a p output"""
        return self.operator.set(self.SET_NAME, value=value, ex=ex, nx=nx)
    
    def get_output_p(self):
        """Get a p output"""
        return self.operator.get(self.SET_NAME)
    
