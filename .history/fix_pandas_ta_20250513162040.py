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
        import pandas_ta
    except ImportError:
        print("Error: numpy or pandas_ta not found. Please install them first.")
        return False
    
    # Get pandas_ta module location
    pandas_ta_dir = os.path.dirname(pandas_ta.__file__)
    core_py_path = os.path.join(pandas_ta_dir, "core.py")
    
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
        print("No import statement found, looking for direct NaN references...")
        # This pattern replaces numpy.NaN with numpy.nan 
        content = re.sub(r'numpy\.NaN', 'numpy.nan', content)
    
    # This pattern replaces bare NaN with numpy.nan
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
        print("No changes were needed.")
        return True

if __name__ == "__main__":
    if fix_pandas_ta():
        print("Patch completed successfully!")
    else:
        print("Patch failed.") 