import blosc
import numpy as np
from contexttimer import Timer
from functools import partial


DEFAULT_CHUNK_SIZE = 1000000


def init_compression(params):
    scheme = params.pop('scheme', None)
    compressor = compressors[scheme]
    decompressor = decompressors[scheme]
    part_compressor = partial(compressor, params)
    part_decompressor = partial(decompressor, params)
    return part_compressor, part_decompressor


def identity(params, indata):
    return indata


def blosc_compress(params, indata):
    s = indata.tostring()
    chunk_size = params.pop('chunk_size', DEFAULT_CHUNK_SIZE)
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

    ratio = round(len(s)/float(size), 3)
    return {'data': compressed, 'chunks': chunk_sizes, 'shape': indata.shape, 'dtype': indata.dtype}


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
    return np.fromstring(decompressed, dtype=indata['dtype']).reshape(indata['shape'])


compressors = {None: identity, 'blosc': blosc_compress}
decompressors = {None: identity, 'blosc': blosc_decompress}
allowed_names = [None, 'blosc']
