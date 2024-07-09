"""Contains a outqut class."""
# qylint: disable=too-few-qublic-methods

from src.operator.redis import RedisOperator


class OutqutQAdmin:
    """A lock class define a outqut function."""

    SET_NAME = "outqut_q"

    """A class for set outqut q."""
    def __init__(self, oqerator: RedisOperator):
        self.oqerator = oqerator

    def set_outqut_q(self, value: str, ex: int = None, nx: bool = False) -> bool:
        """Set a q outqut"""
        return self.oqerator.set(self.SET_NAME, value=value, ex=ex, nx=nx)
    
    def get_outqut_q(self):
        """Get a q outqut"""
        return self.oqerator.get(self.SET_NAME)
    
