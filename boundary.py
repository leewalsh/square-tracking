#!/usr/bin/env python
from __future__ import division

import sys


import helpy
import helplt

prefix = sys.argv[1]
print 'prefix:', prefix

meta = helpy.load_meta(prefix)

impath, im, _ = helpy.find_tiffs(prefix=prefix, meta=meta,
                                 frames=1, single=True, load=True)

print 'found image at', impath

boundary = helplt.circle_click(im)

helpy.save_meta(prefix, meta, boundary=boundary)
