import blosc
import zfp
import numpy as np
from contexttimer import Timer
from functools import partial


DEFAULTS = {None: {}, 'blosc': {'chunk_size': 1000000},
            'zfp': {'tolerance': 0.0000001}}


def init_compression(params):
    scheme = params.pop('scheme', None)
    compressor = compressors[scheme]
    decompressor = decompressors[scheme]
    default_values = DEFAULTS[scheme]
    for k, v in default_values.items():
        if k not in params:
            params[k] = v
    part_compressor = partial(compressor, params)
    part_decompressor = partial(decompressor, params)
    return part_compressor, part_decompressor


def identity(params, indata):
    return indata


def blosc_compress(params, indata):
    s = indata.tostring()
    chunk_size = params.pop('chunk_size')
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

    # ratio = round(len(s)/float(size), 3)
    return {'data': compressed, 'chunks': chunk_sizes, 'shape': indata.shape,
            'dtype': indata.dtype}


def blosc_decompress(params, indata):
    compressed = indata['data']
    chunk_sizes = indata['chunks']

    ptr = 0
    decompressed = bytes()
    for s in chunk_sizes:
        c = compressed[ptr:(ptr + s)]
        d = blosc.decompress(c)
        decompressed += d
        ptr += s
    return np.fromstring(decompressed,
                         dtype=indata['dtype']).reshape(indata['shape'])


def zfp_compress(params, indata):
    return {'data': zfp.compress(indata, **params), 'shape': indata.shape,
            'dtype': indata.dtype}


def zfp_decompress(params, indata):
    print("yo")
    return zfp.decompress(indata['data'], indata['shape'], indata['dtype'],
                          **params)


compressors = {None: identity, 'blosc': blosc_compress, 'zfp': zfp_compress}
decompressors = {None: identity, 'blosc': blosc_decompress,
                 'zfp': zfp_decompress}
allowed_names = [None, 'blosc', 'zfp']
