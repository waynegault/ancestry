#!/usr/bin/env python3

# Test partial utils.py import to isolate hanging issue

print("Starting partial import test...")

# --- Path management and optimization imports ---
from core_imports import standardize_module_imports

print("✅ Imported standardize_module_imports")

# --- Unified import system ---
from core_imports import (
    standardize_module_imports,
    auto_register_module,
    register_function,
    get_function,
    is_function_available,
)

print("✅ Imported core_imports functions")

standardize_module_imports()

print("✅ Called standardize_module_imports()")

# --- Ensure core utility functions are always importable ---
import re
import logging
import time
import json
import requests
import cloudscraper
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)  # Consolidated typing imports

print("✅ Standard imports completed")

# --- Standard library imports ---
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
import contextlib  # <<<< MODIFIED LINE: Added import for contextlib
import json  # For make_ube, _api_req (potential json in csrf)
import base64  # For make_ube
import binascii  # For make_ube
import random  # For make_newrelic, retry_api, DynamicRateLimiter
import uuid  # For make_ube, make_traceparent, make_tracestate
import sqlite3  # For SessionManager._initialize_db_engine_and_session (pragma exception)

print("✅ Extended standard library imports completed")

# --- Type Aliases ---
# Import types needed for type aliases
from requests import Response as RequestsResponse
from selenium.webdriver.remote.webdriver import WebDriver

print("✅ Type alias imports completed")

print("✅ Partial import test completed successfully!")
