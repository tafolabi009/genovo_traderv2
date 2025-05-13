import os
import re
import sys

def fix_pandas_ta():
    """
    Fix pandas_ta compatibility with numpy 2.0 by replacing
    references to NaN with numpy.nan
    """
    try:
        import numpy
        print(f"Using numpy version: {numpy.__version__}")
    except ImportError:
        print("Error: numpy not found. Please install numpy first.")
        return False
    
    try:
        import pandas_ta
        print(f"Found pandas_ta at: {pandas_ta.__file__}")
    except ImportError:
        print("Error: pandas_ta not found. Please install pandas_ta first.")
        print("Run: pip install pandas_ta")
        return False
    except Exception as e:
        print(f"Warning: pandas_ta imported but had an error: {e}")
        # Continue anyway to try fixing the file
    
    # Get pandas_ta module location
    try:
        pandas_ta_dir = os.path.dirname(pandas_ta.__file__)
        core_py_path = os.path.join(pandas_ta_dir, "core.py")
    except Exception as e:
        print(f"Error determining pandas_ta location: {e}")
        # Try to find it another way
        try:
            import site
            site_packages = site.getsitepackages()[0]
            pandas_ta_dir = os.path.join(site_packages, "pandas_ta")
            core_py_path = os.path.join(pandas_ta_dir, "core.py")
        except Exception as e:
            print(f"Error finding pandas_ta in site-packages: {e}")
            return False
    
    if not os.path.exists(core_py_path):
        print(f"Error: pandas_ta core.py not found at: {core_py_path}")
        return False
    
    print(f"Fixing pandas_ta core.py at: {core_py_path}")
    
    # Create backup
    backup_path = core_py_path + ".backup"
    if not os.path.exists(backup_path):
        try:
            with open(core_py_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
            print(f"Backup created at: {backup_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    # Read core.py
    try:
        with open(core_py_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading core.py: {e}")
        return False
    
    # Replace problematic imports
    original_content = content
    
    # Replace import statements
    content = re.sub(r'from\s+numpy\s+import\s+NaN', 'from numpy import nan as NaN', content)
    
    # If no replacement happened, look for direct references
    if content == original_content:
        print("No import statement changed. Looking for direct NaN references...")
        # This pattern replaces numpy.NaN with numpy.nan 
        content = re.sub(r'numpy\.NaN', 'numpy.nan', content)
    
    # Also replace any standalone NaN (not inside variable names)
    content = re.sub(r'(?<!\w)NaN(?!\w)', 'numpy.nan', content)
    
    # Write the modified file
    if content != original_content:
        try:
            with open(core_py_path, 'w') as f:
                f.write(content)
            print("Successfully patched pandas_ta for numpy 2.0 compatibility!")
            return True
        except Exception as e:
            print(f"Error writing modified core.py: {e}")
            return False
    else:
        print("No changes were needed or no NaN references found.")
        
        # Check if there is any other NaN reference causing issues
        try:
            # Create a simple test to see if pandas_ta works with the current numpy
            import pandas as pd
            df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
            # Try a basic function
            print("Testing a basic pandas_ta function...")
            result = df.ta.sma(length=2)
            print("Test successful! pandas_ta works with your numpy version.")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            print("The numpy/pandas_ta compatibility issue might be more complex.")
            return False

if __name__ == "__main__":
    print("Starting pandas_ta fix for numpy 2.0 compatibility...")
    if fix_pandas_ta():
        print("Fix completed successfully!")
    else:
        print("Fix failed. You might need to downgrade numpy to a compatible version.")
        print("Try: pip install numpy==1.24.3 --force-reinstall") 
