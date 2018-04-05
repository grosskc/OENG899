#!/anaconda3/bin/python
import os
import shutil
import sys
import subprocess

if __name__ == '__main__':
    fname, ext = os.path.splitext(sys.argv[1])
    shutil.copy2(fname + ext, fname+ext + ".bak")

with open(fname + ext, 'r') as f:
    fileContentsLines = f.readlines()

codeStart = []
codeStop = []
lineCounter = 0
for line in fileContentsLines:
    if line.strip() == "```python":
        codeStart.append(lineCounter+2)
    elif line.strip() == "```":
        codeStop.append(lineCounter)
    lineCounter += 1

if len(codeStart) == len(codeStop):
    for i, j in zip(codeStart, codeStop):
        print(f"Auto-formatting code block between lines {i} and {j}")
        my_cmd = f"autopep8 --in-place --line-range {i} {j} --max-line-length 100 {sys.argv[1]}"
        sts = subprocess.call(my_cmd.strip(), shell=True)
else:
    print("There is at least one code block that is not properly fenced.")
