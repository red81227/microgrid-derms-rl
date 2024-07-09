"""Contains a Redis Operator class."""
# pylint: disable=no-member
import redis

from config.project_setting import redis_config


class RedisOperator:
    #pylint: disable=too-many-public-methods
    """Operator with Redis backend."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        # pylint: disable=unused-argument
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._set_conn_pool()

    def _set_conn_pool(self):
        connection_info = redis_config.dict()
        # connection_info["password"] = os.environ.get("REDIS_PASSWORD")
        self._redis_conn_pool = redis.ConnectionPool(**connection_info)

    @property
    def redis_conn(self):
        """Return a redis connection object."""
        return redis.Redis(connection_pool=self._redis_conn_pool)
    
    def check_connection(self):
        """Check redis connection."""
        return self.redis_conn.ping()

    def set(self, name: str, value, ex=None, px=None, nx=False, xx=False):
        """Set the value of a key."""
        return self.redis_conn.set(name=name, value=value, ex=ex, px=px, nx=nx, xx=xx)

    def get(self, name: str):
        """Get the value of a key."""
        return self.redis_conn.get(name=name)
    
    def hmset(self, name: str, mapping):
        """Set multiple hash fields to multiple values."""
        return self.redis_conn.hmset(name, mapping)

    def hgetall(self, name: str):
        """Get the values of all the given hash fields."""
        return self.redis_conn.hgetall(name)