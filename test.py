from dataset import AsciiArtDataset
from ascii_util import one_hot_embedded_matrix_to_string

ds = AsciiArtDataset(ragged_batch_bin=True, ragged_batch_bin_batch_size=8, res=128)
print(len(ds))

for i in range(len(ds)):
    s, l = ds[i]
    new_size = s.shape[1]
    print(new_size)
    assert new_size == s.shape[2]

