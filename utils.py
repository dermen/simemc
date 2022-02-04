import numpy as np
from scipy.spatial.transform import Rotation


def load_quat_file(quat_bin):
    """
    Load the data file written by Ti-Yen's quaternion grid sampler
    """
    num_quat = np.fromfile(quat_bin, np.int32, 1)[0]
    quat_data = np.fromfile(quat_bin, np.float64, offset=4)
    quat_data = quat_data.reshape((num_quat, 5))

    # Convert these quats to rotation matrices using scipy
    rotMats = Rotation.from_quat(quat_data[:, :4]).as_matrix()
    weights = quat_data[:,4]

    return rotMats, weights


