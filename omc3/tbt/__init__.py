"""Exposes TbtData, read_tbt and write_tbt directly in tbt namespace."""
from .handler import TbtData, read_tbt, write_tbt

write = write_tbt
read = read_tbt

# Importing * is a bad practice and you should be punished for using it
__all__ = []
