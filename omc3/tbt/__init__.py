"""Exposes TbtData, read_tbt and write_tbt directly in tbt namespace."""
from tbt.handler import TbtData, write_tbt, read_tbt
write = write_tbt
read = read_tbt
