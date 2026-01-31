from pathlib import *
import os
import sys

"""
A python script that build a CUDA program by creating an nvcc call that compiles
each source file in the directory, recursively.
"""

def getAllSourceFiles(dir: Path) -> list[Path]:
    sourceFiles = []
    for item in dir.iterdir():
        if item.name.endswith(".cu") or item.name.endswith(".cpp"):
            sourceFiles.append(item)
        if item.is_dir():
            sourceFiles.extend(getAllSourceFiles(item))
    return sourceFiles

debug = False
if len(sys.argv) > 1:
    debug = True

cwd = Path(".")

sourceFiles = getAllSourceFiles(cwd)

relFileNames = []
fileNamesNoExt = []

for file in sourceFiles:
    relFileNames.append(str(file.relative_to(".")))
    fileNamesNoExt.append(file.name.split(".")[0])

#nvcc -dc src/*.cu main.cu && nvcc *.obj -o main && del *.obj
if (False):
    pass
else:
    command = "nvcc -std=c++20 -dc "

for relFileName in relFileNames:
    command += relFileName + " "

command += "&& nvcc -std=c++20 "

for fileName in fileNamesNoExt:
    command += fileName + ".obj "

command += "-o main && del *.obj"

os.system(command)