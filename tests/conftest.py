"""Define fixtures for pytest."""
from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch
import pandas as pd
from src import create_app


@pytest.fixture(scope="session", name="test_client")
def get_test_client():
    """Return the FastAPI test client, which is shared for each session."""
    app = create_app()
    #app.dependency_overrides[get_current_user_with_correct_group] = override_get_current_user_with_correct_group
    with TestClient(app) as client:
        yield client