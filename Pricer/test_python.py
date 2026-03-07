print("Python is working!")
print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import scipy
    print("✓ scipy imported successfully")
except ImportError as e:
    print(f"✗ scipy import failed: {e}")

try:
    import matplotlib
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    from toolkit import odeRK4
    print("✓ toolkit.odeRK4 imported successfully")
except ImportError as e:
    print(f"✗ toolkit import failed: {e}")

print("\nIf all imports succeeded, you can run pricer_validation.py")
