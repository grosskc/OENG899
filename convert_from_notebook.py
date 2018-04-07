#!/anaconda3/bin/python
"""Tool to convert python cell notebook to a markdown file"""

import os
import shutil
import sys

if __name__ == '__main__':
    # Extract filename and extension, make backup copy
    fname, ext = os.path.splitext(sys.argv[1])
    if ext == ".py":
        try:
            # make backup copy if MD file already exists
            shutil.copy2(fname + ".md", fname + ".md.bak")
        except OSError as e:
            if e.errno == 2:
                # suppress "No such file or directory" error
                pass
            else:
                # reraise the exception, as it's an unexpected error
                raise

        with open(fname + ext, 'r') as f:
            fileContentsLines = f.readlines()

        # Remove "#' " 
        # Also do "#'" in case trailing space was removed
        for i, _ in enumerate(fileContentsLines):
            fileContentsLines[i].lstrip("#' ")
            fileContentsLines[i].lstrip("#'")

        with open(fname + ".md", 'w') as f:
            f.writelines(fileContentsLines)
    else:
        print(f"Cannot convert {fname+ext} from notebook.")
