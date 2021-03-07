#run.py

from _example import ffi, lib

@ffi.callback("int(int, int *)")
def python_callback(how_many, values):
    print (ffi.unpack(values, how_many))
    return 0
lib.python_callback = python_callback
