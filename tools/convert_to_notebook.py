#!/anaconda3/bin/python
"""
Tool to convert a markdown file with python code blocks
into a python notebook
"""

import os
import shutil
import sys

if __name__ == '__main__':

    # Extract filename and extension, make backup copy
    fname, ext = os.path.splitext(sys.argv[1])
    if ext == ".md":
        shutil.copy2(fname + ext, fname+ext + ".bak")

        with open(fname + ext, 'r') as f:
            fileContentsLines = f.readlines()

        # Determine line numbers bracketing code blocks
        codeStart = []
        codeStop = []
        lineCounter = 0
        for line in fileContentsLines:
            if line.strip() == "```python":
                codeStart.append(lineCounter+2)
            elif line.strip() == "```":
                codeStop.append(lineCounter)
            lineCounter += 1

        # Compute which line numbers are not code blocks
        nonCodeStart = [0] + codeStop
        nonCodeStop = codeStart + [len(fileContentsLines)+1]

        # prepend "#' " in front of non-code blocks and add cell marker before each code block
        for i, j in zip(nonCodeStart, nonCodeStop):
            for ix in range(i, j - 2):
                fileContentsLines[ix] = "#' " + fileContentsLines[ix]
            fileContentsLines[j-2] = "#' " + fileContentsLines[j-2] + "\n# %%\n" # make it a cell

        with open(fname + ".py", 'w') as f:
            f.writelines(fileContentsLines)
    else:
        print(f"Cannot convert {fname+ext} from Markdown to Notebook.")
