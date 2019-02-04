"""Exposes TbtData, read_tbt and write_tbt directly in tbt namespace."""
from tbt.handler import read_tbt, write_tbt, TbtData
# aliases
read = read_tbt
write = write_tbt