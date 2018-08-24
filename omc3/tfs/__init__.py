"""Exposes TfsDataFrame, read_tfs and write_tfs directly in tfs namespace."""
from tfs.handler import read_tfs, write_tfs, TfsDataFrame
# aliases
read = read_tfs
write = write_tfs
