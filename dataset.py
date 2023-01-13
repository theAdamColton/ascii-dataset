import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from os import path
import random
from glob import glob
from tqdm import tqdm

from torchdata.datapipes.iter import IterableWrapper


DATADIR = path.abspath(path.join(path.dirname(__file__), "data_aggregation/data/"))
import ascii_util
import string_utils



class AsciiArtDataset(Dataset):
    """
    Initialize an AsciiArtDataset, from a directory structure where subfolders are the sub categories
    And ascii art files are stored as .txt ascii files. If `res` is passed, all arts larger (greater #
    of lines, or greater # of columns) than this value are not included in the dataset. Ascii files
    smaller than res will be padded with spaces.

    Ascii art files in the datapath are expected to already be pre padded to the right with zeros to
    the length of the maximum width line.

    Files are returned as tensors, with each character being represented by a vector embedding.
    These embeddings are 'one-hot' vectors of length 95.
    The only characters that are allowed are 32 to 126 (inclusive).
    """

    def __init__(
        self,
        res: int = 36,
        datapath: str = None,
        max_samples=None,
        validation_prop=None,
        is_validation_dataset=False,
        ragged_batch_bin=False,
        ragged_batch_bin_batch_size=None,
    ):
        """
        res: Maximum resolution of the square ascii art
        ragged_batch_bin: Allows different batches to have different resolutions. Across a single batch, all images will have the same batch size.
            Use this with ragged_batch_bin_batch_size.
            This argument will cause this dataset to return batches, not individual data points
        datapath: Optional specification of the directory containing *.txt files, organized by directory in categories
        max_samples: The maximum number of training samples to take.
        validation_prop: The proportion of data to use as validation
        is_validation_dataset: If this is true, this dataset will only return items from the validation dataset
        """
        self.res = res

        self.channels = 95

        if not datapath:
            datapath = DATADIR

        assert path.isdir(datapath)
        self.datapath = datapath
        # Filters out files that are too large
        asciilengths = dict()
        for file in glob(path.join(datapath, "**/*.txt"), recursive=True):
            with open(file, "r") as f:
                line_count = sum(1 for _ in f.readlines())
            with open(file, "r") as f:
                line1 = f.readline()
                line_width = len(line1)
            side_res = max(line_count, line_width)
            if res is not None:
                if side_res > res:
                    continue
            asciilengths[file] = side_res

        asciifiles = list(asciilengths.keys())
        # Sorts by length
        asciifiles.sort(key=lambda x: asciilengths[x])
        self.asciifiles = asciifiles

        self.ragged_batch_bin = ragged_batch_bin
        self.ragged_batch_bin_batch_size = ragged_batch_bin_batch_size
        if ragged_batch_bin:
            assert ragged_batch_bin_batch_size

            self.pad_to_size = dict()

            # Fills out self.pad_to_size such that every file in a batch has the same size,
            # that being the largest resolution ascii file in the batch.
            for batch_i in range(0, len(asciifiles), ragged_batch_bin_batch_size):
                max_side_res = -1
                for j in range(ragged_batch_bin_batch_size):
                    if batch_i + j >= len(asciifiles):
                        break
                    file = asciifiles[batch_i + j]
                    side_res = asciilengths[file]
                    if side_res > max_side_res:
                        max_side_res = side_res

                for j in range(ragged_batch_bin_batch_size):
                    if batch_i + j >= len(asciifiles):
                        break
                    file = asciifiles[batch_i + j]
                    self.pad_to_size[file] = max_side_res


        if validation_prop:
            max_idx = int(len(self.asciifiles) * (1 - validation_prop))
            self.validation_ascii_files = self.asciifiles[max_idx:]
            self.asciifiles = self.asciifiles[0:max_idx]
            print(
                "#{} training files, #{} validation files".format(
                    len(self.asciifiles), len(self.validation_ascii_files)
                )
            )
        if is_validation_dataset:
            self.asciifiles = self.validation_ascii_files
        if max_samples:
            self.asciifiles = self.asciifiles[: max_samples + 1]

    def __len__(self):
        if not self.ragged_batch_bin:
            return len(self.asciifiles)
        else:
            return len(self.asciifiles) // self.ragged_batch_bin_batch_size

    def __getitem__(self, index):
        """
        Returns the character_embeddings representation of the string,
        as a self.channels by self.res by self.res array
        """
        if not self.ragged_batch_bin_batch_size:
            filename = self.asciifiles[index]
            return self.__getitem_from_filename__(filename)
        else:
            start_idx = index * self.ragged_batch_bin_batch_size
            batch_ascii_res = self.pad_to_size[self.asciifiles[start_idx]]
            batch_out = np.empty((self.ragged_batch_bin_batch_size, 95, batch_ascii_res, batch_ascii_res,)) 
            batch_out_labels = [""] * self.ragged_batch_bin_batch_size

            def f(i):
                item = self.__getitem_from_filename__(self.asciifiles[i])[0]
                return item

            for i in range(start_idx, start_idx + self.ragged_batch_bin_batch_size):
                batch_out[i-start_idx] = f(i)
                batch_out_labels[i-start_idx] = self.__get_category_string_from_datapath(self.asciifiles[i])
            return batch_out, batch_out_labels
            


    def get_validation_item(self, index):
        filename = self.validation_ascii_files[index]
        return self.__getitem_from_filename__(filename)

    def get_random_training_item(self):
        filename = self.asciifiles[random.randint(0, len(self.asciifiles) - 1)]
        return self.__getitem_from_filename__(filename)

    def get_validation_length(self):
        return len(self.validation_ascii_files)

    def __getitem_from_filename__(self, filename):
        with open(filename, "r") as f:
            content = f.read()

        content = ascii_util.pad_to_x_by_x(content, self.pad_to_size[filename])
        embeddings = ascii_util.any_shape_string_to_one_hot(content)

        label = self.__get_category_string_from_datapath(filename)

        return embeddings, label

    def to_tensordataset(self, device) -> TensorDataset:
        out = torch.Tensor(
            len(self),
            self.channels,
            self.res,
            self.res,
        ).to(device)
        for i in range(len(self)):
            out[i] = torch.Tensor(self[i][0]).to(device)
        return TensorDataset(out)

    def get_all_category_strings(self):
        """Returns all category strings,unordered"""
        d = set()
        for x in self.asciifiles:
            d.add(self.__get_category_string_from_datapath(x))
        return list(d)

    def __get_category_string_from_datapath(self, datapath: str) -> str:
        return string_utils.remove_prefix(path.dirname(datapath), self.datapath)

    def decode(self, x) -> str:
        """Takes a matrix of character embeddings, returns a string with correct line breaks"""
        if not type(x) == np.ndarray:
            x = x.cpu()
            x = np.array(x)

        return ascii_util.one_hot_embedded_matrix_to_string(x)

    def get_file_name(self, i):
        return self.asciifiles[i]

    def calculate_character_counts(self):
        """
        Goes through every character in every artwork in the training dataset and counts it

        Returns a 95 length tensor of the character counts
        """

        counts = torch.zeros(95, dtype=torch.int32)

        print("Loading character frequencies")
        for x, label in tqdm(self):
            x = torch.IntTensor(x)
            character_indeces = x.argmax(0)
            x_char_counts = torch.bincount(character_indeces.flatten(), minlength=95)
            counts += x_char_counts

        return counts

