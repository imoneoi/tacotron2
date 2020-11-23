import re
from text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
  # Split text by space
  sequence = text.split(" ")

  # Convert symbol to id
  sequence = [_symbol_to_id[item] for item in sequence]

  return sequence


def sequence_to_text(sequence):
  # Convert from id to symbol
  text = [_id_to_symbol[item] for item in sequence]

  # Concat by space
  text = " ".join(text)

  return text