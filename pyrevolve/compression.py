import blosc
import numpy as np
from contexttimer import Timer


CHUNK_SIZE = 1000000


def init_compression(params):
    global CHUNK_SIZE
    chunk = params.get('chunk_size', None)
    if chunk is not None:
        CHUNK_SIZE = chunk


def identity(indata):
    return indata


def blosc_compress(indata):
    s = indata.tostring()
    global CHUNK_SIZE
    chunked = [s[i:i+CHUNK_SIZE] for i in range(0, len(s), CHUNK_SIZE)]
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
    print("Compression Time: %d, ratio: %f, chunk size: %d" % (time, ratio,
                                                               CHUNK_SIZE))
    return {'data': compressed, 'chunks': chunk_sizes, 'shape': indata.shape, 'dtype': indata.dtype}


def blosc_decompress(indata):
    global CHUNK_SIZE
    compressed = indata['data']
    chunk_sizes = indata['chunks']
    
    ptr = 0
    decompressed = bytes()
    for s in chunk_sizes:
        c = compressed[ptr:(ptr + s)]
        d = blosc.decompress(c)
        decompressed += d
        ptr += s
    # s = indata.tostring()
    # with Timer as t:
    #    c = s.compress()

    # ratio = round(len(s)/float(len(c)), 3)
    # print("Decompression Time: %d, ratio: %f" % (t.elapsed, ratio))
    return np.fromstring(decompressed, dtype=indata['dtype']).reshape(indata['shape'])


compressors = {None: identity, 'blosc': blosc_compress}
decompressors = {None: identity, 'blosc': blosc_decompress}
allowed_names = [None, 'blosc']
