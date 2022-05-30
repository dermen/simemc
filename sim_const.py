
from dxtbx.model import Detector, Beam
from dxtbx.model.crystal import CrystalFactory


BEAMSIZE=0.01  # mm, size of beam
NCELLS_ABC=30,30,40  # mosaic domain size in unit cell lengths
NCELLS_ABC=90,21,24  # mosaic domain size in unit cell lengths
OVERSAMPLE=3  # pixel sub-sampling factor
MOS_DOMS=1  # makes sim slow if this is high, but used to generate spherical cap profiles (100-500 depending on mos_spread)
MOS_SPREAD=0  # degrees, angular mosaicity if mos_doms > 1. Larger values require more domains (0.01-0.07 ish)
PROFILE="gauss"  # square, round, tophat or gauss
#TOTAL_FLUX=1e10
TOTAL_FLUX=5e9  # number of photons per pulse
WATER_PATH=1  # mm

# determines range of spots that will be on detector
DMIN=1.4  # angstrom
DMAX=30 # angstrom


########
# GEOM
_DET_DICT= {
    'panels': [{'name': 'Panel',
        'type': 'SENSOR_PAD',
        'fast_axis': (1.0, 0.0, 0.0),
        'slow_axis': (0.0, -1.0, 0.0),
        'origin': (-216.892, 221.192, -450.0),
        'raw_image_offset': (0, 0),
        'image_size': (2463, 2527),
        'pixel_size': (0.17200000000000001, 0.17200000000000001),
        'trusted_range': (-1.0, 960687.0),
        'thickness': 0.0,
        'material': 'Si',
        'mu': 0.0,
        'identifier': '',
        'mask': [(487, 0, 494, 2527),
                 (981, 0, 988, 2527),
                 (1475, 0, 1482, 2527),
                 (1969, 0, 1976, 2527),
                 (0, 195, 2463, 212),
                 (0, 407, 2463, 424),
                 (0, 619, 2463, 636),
                 (0, 831, 2463, 848),
                 (0, 1043, 2463, 1060),
                 (0, 1255, 2463, 1272),
                 (0, 1467, 2463, 1484),
                 (0, 1679, 2463, 1696),
                 (0, 1891, 2463, 1908),
                 (0, 2103, 2463, 2120),
                 (0, 2315, 2463, 2332)],
        'gain': 1.0,
        'pedestal': 0.0,
        'px_mm_strategy': {'type': 'SimplePxMmStrategy'}}],
    'hierarchy': {'name': '',
          'type': '',
          'fast_axis': (1.0, 0.0, 0.0),
          'slow_axis': (0.0, 1.0, 0.0),
          'origin': (0.0, 0.0, 0.0),
          'raw_image_offset': (0, 0),
          'image_size': (0, 0),
          'pixel_size': (0.0, 0.0),
          'trusted_range': (0.0, 0.0),
          'thickness': 0.0,
          'material': '',
          'mu': 0.0,
          'identifier': '',
          'mask': [],
          'gain': 1.0,
          'pedestal': 0.0,
          'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
          'children': [{'panel': 0}]}}


_BEAM_DICT = {
    'direction': (0.0, 0.0, 1.0),
    'wavelength': 1.03324,
    'divergence': 0.0,
    'sigma_divergence': 0.0,
    'polarization_normal': (0.0, 1.0, 0.0),
    'polarization_fraction': 0.999,
    'flux': 0.0,
    'transmission': 1.0}

UCELL_A=79.1
UCELL_B=79.1
UCELL_C=38.4
HALL='-P 4 2'

UCELL_A = 68
UCELL_B = 68
UCELL_C = 104
HALL='-P 4 2'

UCELL_A=40.3
UCELL_B=180.3
UCELL_C=142.6
HALL=' C 2c 2'

_CRYSTAL_DICT={
    '__id__': 'crystal',
    'real_space_a': (UCELL_A, 0.0, 0.0),
    'real_space_b': (0.0, UCELL_B, 0.0),
    'real_space_c': (0.0, 0.0, UCELL_C),
    'space_group_hall_symbol': HALL}

DETECTOR = Detector.from_dict(_DET_DICT)
BEAM = Beam.from_dict(_BEAM_DICT)
CRYSTAL = CrystalFactory.from_dict(_CRYSTAL_DICT)
