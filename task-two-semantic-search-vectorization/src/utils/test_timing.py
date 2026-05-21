import logging
import time
from src.utils.timing import log_timing

# Set up logging so we see the output
logging.basicConfig(level=logging.INFO)

@log_timing("test_operation")
def slow_function():
    time.sleep(0.1)
    return 42

# Test it
result = slow_function()
print(f"Result: {result}")
print("✓ Timing decorator works!")