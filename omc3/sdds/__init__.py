"""Exposes SddsFile, read_sdds and write_sdds directly in sdds namespace."""
from sdds.handler import read_sdds, write_sdds, SddsFile
# aliases
read = read_sdds
write = write_sdds
