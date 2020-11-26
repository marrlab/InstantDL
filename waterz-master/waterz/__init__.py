from __future__ import absolute_import
from .evaluate import evaluate

__version__ = '0.8'

def agglomerate(
        affs,
        thresholds,
        gt = None,
        fragments = None,
        aff_threshold_low  = 0.0001,
        aff_threshold_high = 0.9999,
        return_merge_history = False,
        return_region_graph = False,
        scoring_function = 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
        discretize_queue = 0,
        force_rebuild = False):
    '''
    Compute segmentations from an affinity graph for several thresholds.

    Passed volumes need to be converted into contiguous memory arrays. This will
    be done for you if needed, but you can save memory by making sure your
    volumes are already C_CONTIGUOUS.

    Parameters
    ----------

        affs: numpy array, float32, 4 dimensional

            The affinities as an array with affs[channel][z][y][x].

        thresholds: list of float32

            The thresholds to compute segmentations for. For each threshold, one
            segmentation is returned.

        gt: numpy array, uint32, 3 dimensional (optional)

            An optional ground-truth segmentation as an array with gt[z][y][x].
            If given, metrics

        fragments: numpy array, uint64, 3 dimensional (optional)

            An optional volume of fragments to use, instead of the build-in 
            zwatershed.

        aff_threshold_low: float, default 0.0001
        aff_threshold_high: float, default 0.9999,

            Thresholds on the affinities for the initial segmentation step.

        return_merge_history: bool

            If set to True, the returning tuple will contain a merge history,
            relative to the previous segmentation.

        return_region_graph: bool

            If set to True, the returning tuple will contain the region graph
            for the returned segmentation.

        scoring_function: string, default 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>'

            A C++ type string specifying the edge scoring function to use. See

                https://github.com/funkey/waterz/blob/master/waterz/backend/MergeFunctions.hpp

            for available functions, and

                https://github.com/funkey/waterz/blob/master/waterz/backend/Operators.hpp

            for operators to combine them.

        discretize_queue: int

            If set to non-zero, a bin queue with that many bins will be used to 
            approximate the priority queue for merge operations.

        force_rebuild:

            Force the rebuild of the module. Only needed for development.

    Returns
    -------

        Results are returned as tuples from a generator object, and only
        computed on-the-fly when iterated over. This way, you can ask for
        hundreds of thresholds while at any point only one segmentation is
        stored in memory.

        Depending on the given parameters, the returned values are a subset of
        the following items (in that order):

        segmentation

            The current segmentation (numpy array, uint64, 3 dimensional).

        metrics (only if ground truth was provided)

            A  dictionary with the keys 'V_Rand_split', 'V_Rand_merge',
            'V_Info_split', and 'V_Info_merge'.

        merge_history (only if return_merge_history is True)

            A list of dictionaries with keys 'a', 'b', 'c', and 'score',
            indicating that region a got merged with b into c with the given
            score.

        region_graph (only if return_region_graph is True)

            A list of dictionaries with keys 'u', 'v', and 'score', indicating
            an edge between u and v with the given score.

    Examples
    --------

        affs = ...
        gt   = ...

        # only segmentation
        for segmentation in agglomerate(affs, range(100,10000,100)):
            # ...

        # segmentation with merge history
        for segmentation, merge_history in agglomerate(affs, range(100,10000,100), return_merge_history = True):
            # ...

        # segmentation with merge history and metrics compared to gt
        for segmentation, metrics, merge_history in agglomerate(affs, range(100,10000,100), gt, return_merge_history = True):
            # ...
    '''

    import sys, os
    import shutil
    import glob
    import numpy
    import fcntl

    try:
        import hashlib
    except ImportError:
        import md5 as hashlib

    from distutils.core import Distribution, Extension
    from distutils.command.build_ext import build_ext
    from distutils.sysconfig import get_config_vars, get_python_inc

    import Cython
    from Cython.Compiler.Main import Context, default_options
    from Cython.Build.Dependencies import cythonize

    # compile agglomerate.pyx for given scoring function

    source_dir = os.path.dirname(os.path.abspath(__file__))
    source_files = [
            os.path.join(source_dir, 'agglomerate.pyx'),
            os.path.join(source_dir, 'frontend_agglomerate.h'),
            os.path.join(source_dir, 'frontend_agglomerate.cpp')
    ]
    source_files += glob.glob(source_dir + '/backend/*.hpp')
    source_files.sort()
    source_files_hashes = [ hashlib.md5(open(f, 'r').read().encode('utf-8')).hexdigest() for f in source_files ]

    key = scoring_function, discretize_queue, source_files_hashes, sys.version_info, sys.executable, Cython.__version__
    module_name = 'waterz_' + hashlib.md5(str(key).encode('utf-8')).hexdigest()
    lib_dir=os.path.expanduser('~/.cython/inline')

    # since this could be called concurrently, there is no good way to check
    # whether the directory already exists
    try:
        os.makedirs(lib_dir)
    except:
        pass

    # make sure the same module is not build concurrently
    with open(os.path.join(lib_dir, module_name + '.lock'), 'w') as lock_file:
        fcntl.lockf(lock_file, fcntl.LOCK_EX)

        try:

            if lib_dir not in sys.path:
                sys.path.append(lib_dir)
            if force_rebuild:
                raise ImportError
            else:
                __import__(module_name)

            print("Re-using already compiled waterz version")

        except ImportError:

            print("Compiling waterz in " + str(lib_dir))

            cython_include_dirs = ['.']
            ctx = Context(cython_include_dirs, default_options)

            include_dir = os.path.join(lib_dir, module_name)
            if not os.path.exists(include_dir):
                os.makedirs(include_dir)

            include_dirs = [
                source_dir,
                include_dir,
                os.path.join(source_dir, 'backend'),
                os.path.dirname(get_python_inc()),
                numpy.get_include(),
            ]

            scoring_function_header = os.path.join(include_dir, 'ScoringFunction.h')
            with open(scoring_function_header, 'w') as f:
                f.write('typedef %s ScoringFunctionType;'%scoring_function)

            queue_header = os.path.join(include_dir, 'Queue.h')
            with open(queue_header, 'w') as f:
                if discretize_queue == 0:
                    f.write('template<typename T, typename S> using QueueType = PriorityQueue<T, S>;')
                else:
                    f.write('template<typename T, typename S> using QueueType = BinQueue<T, S, %d>;'%discretize_queue)

            # cython requires that the pyx file has the same name as the module
            shutil.copy(
                    os.path.join(source_dir, 'agglomerate.pyx'),
                    os.path.join(lib_dir, module_name + '.pyx')
            )
            shutil.copy(
                    os.path.join(source_dir, 'frontend_agglomerate.cpp'),
                    os.path.join(lib_dir, module_name + '_frontend_agglomerate.cpp')
            )

            # Remove the "-Wstrict-prototypes" compiler option, which isn't valid 
            # for C++.
            cfg_vars = get_config_vars()
            if "CFLAGS" in cfg_vars:
                cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

            extension = Extension(
                    module_name,
                    sources = [
                        os.path.join(lib_dir, module_name + '.pyx'),
                        os.path.join(lib_dir, module_name + '_frontend_agglomerate.cpp')
                    ],
                    include_dirs=include_dirs,
                    language='c++',
                    extra_link_args=['-std=c++11'],
                    extra_compile_args=['-std=c++11', '-w']
            )
            build_extension = build_ext(Distribution())
            build_extension.finalize_options()
            build_extension.extensions = cythonize([extension], quiet=True, nthreads=2)
            build_extension.build_temp = lib_dir
            build_extension.build_lib  = lib_dir
            build_extension.run()

    return __import__(module_name).agglomerate(
        affs,
        thresholds,
        gt,
        fragments,
        aff_threshold_low,
        aff_threshold_high,
        return_merge_history,
        return_region_graph)
