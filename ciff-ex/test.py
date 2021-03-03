import cffi
print(cffi.__version__)
print(cffi.FFI)

ffi = cffi.FFI()
ffi.cdef(
"""
    int printf(const char *format, ...);
""")
lib = ffi.dlopen(None)

arg = ffi.new("char[]", b"world")      
lib.printf(b"hi there, %s.\n", arg)






from  _example import lib , ffi
print(dir(lib))
print(dir(ffi)) 
