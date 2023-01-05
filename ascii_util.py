"""
Utilities for dealing with ascii art
"""

import numpy as np
import torch

import one_hot_encoding
from string_utils import remove_suffix, ljust


def pad_to_max_line_length(s: str, char=" ") -> str:
    """Pads each line of s to the max line length of all the lines in s.
    char: character to pad with
    """
    maxlen = 0
    for l in s.splitlines():
        length = len(l)
        if length > maxlen:
            maxlen = length

    out = ""
    for l in s.splitlines():
        # Gets rid of the last '\n'
        line = remove_suffix(l, "\n")
        padded_line = ljust(line, maxlen, char)
        out += padded_line + "\n"

    return out


def pad_to_x_by_x(s: str, x: int, char=" ") -> str:
    """
    Pads ascii by centering it with ' ' chars
    Assumes that each line of ascii is already padded to the max
    length of its lines
    """
    lines = s.splitlines()
    line_width = len(lines[0])
    assert line_width <= x

    # Vertical padding
    total_vert_padding = x - len(lines)
    assert total_vert_padding >= 0
    assert total_vert_padding <= x
    toppad = total_vert_padding // 2
    botpad = total_vert_padding - toppad

    out = ""
    if toppad != 0:
        out = vertical_pad(x, toppad, char=char) + "\n"
    out += "".join(line.replace("\n", "").center(x, char) + "\n" for line in lines)
    if botpad != 0:
        out += vertical_pad(x, botpad, char=char)

    return out


def vertical_pad(width: int, height: int, char=" ") -> str:
    if height == 0:
        return ""
    out = char * width
    out += ("\n" + char * width) * (height - 1)
    return out


def string_reshape(s: str, x: int) -> str:
    """
    Adds line breaks to s so it becomes a x by y string where y is any
    size
    """
    assert len(s) % x == 0
    res = "\n".join(s[i : i + x] for i in range(0, len(s), x))
    return res


def horizontal_concat(s1: str, s2: str, separator="   |   ") -> str:
    """Concats two 'square' shaped strings"""
    out = ""
    for i, (line1, line2) in enumerate(zip(s1.split("\n"), s2.split("\n"))):
        if i == len(line1) - 1:
            out += line1 + separator + line2
        else:
            out += line1 + separator + line2 + "\n"
    return out


"""                Numpy operations                         """

def any_shape_string_to_one_hot(s: str) -> np.ndarray:
    """
    s has to be rectangular
    """
    height = s.count("\n")
    s_split = s.splitlines(False)
    width = len(s_split[0])
    s_flat = "".join(s_split)
    embedded =one_hot_encoding.get_one_hot_for_str(s_flat)
    if embedded.shape[0] % height != 0:
        import bpdb
        bpdb.set_trace()
    embedded = embedded.reshape(height, width, 95)
    embedded = np.moveaxis(embedded, 2, 0)
    embedded = embedded.astype(np.float32)
    return embedded

def squareized_string_to_one_hot(s: str, x: int) -> np.ndarray:
    """Takes a squareized string s and a length x,
    returns an 95 by x by x array of one hot encodings"""
    s = s.replace("\n", "")
    embedded = one_hot_encoding.get_one_hot_for_str(s)
    embedded = embedded.reshape(x, x, 95)
    # Makes embeddings nchannels by image_res by image_res
    embedded = np.moveaxis(embedded, 2, 0)
    embedded = embedded.astype(np.float32)
    return embedded


def one_hot_embedded_matrix_to_string(a) -> str:
    """Takes a 95 by x by y matrix a of one hot character embeddings and
    returns a string"""
    if type(a) == torch.Tensor:
        a = a.cpu().numpy()
    res_x, res_y = a.shape[1], a.shape[2]
    # Moves channels to last dim
    a = np.moveaxis(a, 0, 2)
    # Flattens
    a = a.reshape(res_x * res_y, 95)
    flat_s = one_hot_encoding.fuzzy_one_hot_to_str(a)
    s = string_reshape(flat_s, res_y)
    return s
