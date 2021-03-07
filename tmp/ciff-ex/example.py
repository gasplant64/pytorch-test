#file "example_build.py"

import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef("""
    int (*python_callback)(int how_many, int *values);
        void *const c_callback;   /* pass this const ptr to C routines */
        """)
ffibuilder.set_source("_example", 
r"""
    #include <stdarg.h>
    #include <alloca.h>
    static int (*python_callback)(int how_many, int *values);
    static int c_callback(int how_many, ...) {
        va_list ap;
        /* collect the "..." arguments intothe values[] array */
        int i, *values = alloca(how_many * sizeof(int));
        va_start(ap, how_many);
        for (i=0; i<how_many; i++)
            values[i] = va_arg(ap, int);
        va_end(ap);
        return python_callback(how_many, values);
                                                   }
""")
ffibuilder.compile(verbose=True)
