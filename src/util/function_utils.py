"""This file is for utility function"""
from functools import lru_cache
from datetime import datetime
import tzlocal


LOCAL_TIMEZONE = tzlocal.get_localzone()

@lru_cache(maxsize=128, typed=False)
def health_check_parsing() -> str:
    """This function is for parsing version information
    Return:
        version_str: str, return the project_version of the setup.py ex: 0.0.1
    """
    with open('setup.py', 'r') as f:
        setup_str = f.read()
    match_str = 'project_version='
    start_pos = setup_str.find(match_str) + len(match_str)
    end_pos = setup_str.find(',', start_pos)
    version_str = setup_str[start_pos:end_pos].replace("'", '')
    return version_str

def get_current_timestamp() -> int:
    """Get current UNIX timestamp in `int` type."""
    return int(datetime.now().timestamp())
