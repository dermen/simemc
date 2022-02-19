"""
dragonfly simulation geometry
"""
from dxtbx.model import Detector, Beam
px = 0.512
n = 150

det_descr = {'panels':
               [{'fast_axis': (1.0, 0.0, 0.0),
                 'gain': 1.0,
                 'identifier': '',
                 'image_size': (n,n),
                 'mask': [],
                 'material': '',
                 'mu': 0.0,
                 'name': 'Panel',
                 'origin': (-px*(n/2.), -px*(n/2.), -300.),
                 'pedestal': 0.0,
                 'pixel_size': (px, px),
                 'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
                 'raw_image_offset': (0, 0),
                 'slow_axis': (0.0, 1.0, 0.0),
                 'thickness': 0.0,
                 'trusted_range': (0.0, 65536.0),
                 'type': ''}]}

DET = Detector.from_dict(det_descr)

beam_descr = {'direction': (0.0, 0.0, 1.0),
             'divergence': 0.0,
             'flux': 1e12,
             'polarization_fraction': 1.,
             'polarization_normal': (0.0, 1.0, 0.0),
             'sigma_divergence': 0.0,
             'transmission': 1.0,
             'wavelength': 6.2}

BEAM = Beam.from_dict(beam_descr)


