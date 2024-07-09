"""This file define the api of service health check"""

from fastapi import APIRouter, HTTPException, status
from config.logger_setting import log
from src.operator.redis import RedisOperator
import redis

from src.util import function_utils
from src.data_models.health_check import HealthCheckBaseModel
# pylint: disable=W0612


def create_health_check_router():
    """This function is for creating health check router"""
    router = APIRouter()

    @router.get("/health_check", response_model=HealthCheckBaseModel, status_code=200)
    def health_check() -> HealthCheckBaseModel:
        """This method is for health check"""
        try:
            redis_operator = RedisOperator()
            _ = redis_operator.check_connection()
            version_str = function_utils.health_check_parsing()
            response = HealthCheckBaseModel(
                version=version_str, msg="Service Up", error="")
            return response
        except redis.ConnectionError:
            error_message = "Redis connection failed"
            log.error(error_message)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_message)
        except Exception as e:
            log.error(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="interal server error")

    return router
