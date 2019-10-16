import h5py
import numpy as np

S = np.sin(np.linspace(-np.pi, np.pi, 2000, endpoint=False))
C = np.cos(np.linspace(-np.pi, np.pi, 2000, endpoint=False))
E = np.exp(-np.linspace(0, 20, 2000, endpoint=False))

with h5py.File(f'test_file_v2.hdf5', 'w') as hd5_file:
    hd5_file.create_group('A1C')
    hd5_file['A1C'].create_dataset("Horizontal", data=S)
    hd5_file['A1C'].create_dataset("Vertical", data=C)
    hd5_file['A1C'].create_dataset("Intensity", data=E)

    hd5_file.create_group('E2R')
    hd5_file['E2R'].create_dataset("Horizontal", data=S)
    hd5_file['E2R'].create_dataset("Vertical", data=C)
    hd5_file['E2R'].create_dataset("Intensity", data=E)
