import sys
print(f"sys.path: {sys.path}")
try:
    import logging
    print(f"logging: {logging}")
    print(f"logging file: {logging.__file__}")
    import logging.config
    print("logging.config imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
