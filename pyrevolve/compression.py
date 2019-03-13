import blosc
import pyzfp
import numpy as np
from contexttimer import Timer
from functools import partial
import pickle


DEFAULTS = {None: {}, 'blosc': {'chunk_size': 1000000},
            'zfp': {'tolerance': 0.0000001, 'parallel': True}}


def init_compression(params):
    params = params.copy()
    scheme = params.pop('scheme', None)
    if scheme == 'custom':
        compressor = params.pop('compressor', None)
        decompressor = params.pop('decompressor', None)
    else:
        compressor = compressors[scheme]
        decompressor = decompressors[scheme]
        default_values = DEFAULTS[scheme]
        for k, v in default_values.items():
            if k not in params:
                params[k] = v
    part_compressor = partial(compressor, params)
    part_decompressor = partial(decompressor, params)
    return part_compressor, part_decompressor


def no_compression_in(params, indata):
    return CompressedObject(memoryview(indata.tobytes()), shape=indata.shape,
                            dtype=indata.dtype)


def no_compression_out(params, indata):
    return np.frombuffer(indata.data, dtype=indata.dtype).reshape(indata.shape)


def blosc_compress(params, indata):
    s = indata.tostring()
    chunk_size = params.get('chunk_size')
    chunked = [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
    time = 0
    size = 0
    compressed = bytes()
    chunk_sizes = []
    for chunk in chunked:
        with Timer(factor=1000) as t:
            c = blosc.compress(chunk)
        compressed += c
        time += t.elapsed
        size += len(c)
        chunk_sizes.append(len(c))
    metadata = {'shape': indata.shape, 'dtype': indata.dtype,
                'chunks': chunk_sizes}
    return CompressedObject(data=compressed, metadata=metadata)


def blosc_decompress(params, indata):
    compressed = indata.data
    chunk_sizes = indata.metadata['chunks']

    ptr = 0
    decompressed = bytes()
    for s in chunk_sizes:
        c = compressed[ptr:(ptr + s)]
        d = blosc.decompress(c)
        decompressed += d
        ptr += s
    return np.frombuffer(decompressed,
                         dtype=indata.dtype).reshape(indata.shape)


class CompressedObject(object):
    def __init__(self, data, shape=None, dtype=None, metadata=None):
        assert(metadata is None or (shape is None and dtype is None))
        if metadata is not None:
            assert('shape' in metadata and 'dtype' in metadata)
            shape = metadata['shape']
            dtype = metadata['dtype']
        else:
            metadata = {'shape': shape, 'dtype': dtype}
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.metadata = metadata
        self.pickled_metadata = pickle.dumps(self.metadata)


def zfp_compress(params, indata):
    return CompressedObject(memoryview(pyzfp.compress(indata, **params)),
                            shape=indata.shape, dtype=indata.dtype)


def zfp_decompress(params, indata):
    assert(isinstance(indata, CompressedObject))
    return pyzfp.decompress(indata.data, indata.shape, indata.dtype,
                            **params)


compressors = {None: no_compression_in, 'blosc': blosc_compress,
               'zfp': zfp_compress}
decompressors = {None: no_compression_out, 'blosc': blosc_decompress,
                 'zfp': zfp_decompress}
allowed_names = [None, 'blosc', 'zfp']
