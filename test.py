from dataset import AsciiArtDataset
from ascii_util import one_hot_embedded_matrix_to_string

ds = AsciiArtDataset(should_pad_to_res=False)

for s, l in ds:
    print(one_hot_embedded_matrix_to_string(s),"\n", l)
