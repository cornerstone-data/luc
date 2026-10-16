import os

os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "2")
# NB: By default, OpenBLAS (what numpy on Linux uses) spawns a thread per core on certain
# operations.  This occurs for every dask thread and can easily saturate CPU's
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
