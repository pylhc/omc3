import h5py
import numpy as np

S = np.sin(np.linspace(-np.pi, np.pi, 2000, endpoint=False))
C = np.cos(np.linspace(-np.pi, np.pi, 2000, endpoint=False))
E = np.exp(-np.linspace(0, 20, 2000, endpoint=False))

with h5py.File(f'test_file.hdf5', 'w') as hd5_file:
    hd5_file.create_dataset("N:IBE2RH", data=S)
    hd5_file.create_dataset("N:IBE2RV", data=C)
    hd5_file.create_dataset("N:IBE2RS", data=E)

    hd5_file.create_dataset("N:IBA1CH", data=S)
    hd5_file.create_dataset("N:IBA1CV", data=C)
    hd5_file.create_dataset("N:IBA1CS", data=E)
