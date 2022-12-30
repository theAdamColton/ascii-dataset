import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from os import path
import random
from glob import glob
from tqdm import tqdm

DATADIR = path.abspath(path.join(path.dirname(__file__), "data_aggregation/data/"))
import utils
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
    ):
        """
        res: Desired resolution of the square ascii art
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
        asciifiles = set(glob(path.join(datapath, "**/*.txt"), recursive=True))
        for file in list(asciifiles).copy():
            with open(file, "r") as f:
                line_count = sum(1 for _ in f)
            with open(file, "r") as f:
                line1 = f.readline()
                line_width = len(line1)
            if res is not None:
                if line_width > res or line_count > res:
                    asciifiles.remove(file)
                    #print("popped {}, too big".format(file))
                    continue

            with open(file, "r") as f:
                for line in f:
                    for s in line:
                        # Only characters in 10, [32, 126] are allowed
                        code = ord(s)
                        if code != 10 and (code < 32 or code > 126):
                            if file in asciifiles:
                                asciifiles.remove(file)

        self.asciifiles = list(asciifiles)
        self.asciifiles.sort()
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
        return len(self.asciifiles)

    def __getitem__(self, index):
        """
        Returns the character_embeddings representation of the string,
        as a self.channels by self.res by self.res array
        """
        filename = self.asciifiles[index]
        return self.__getitem_from_filename__(filename)

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

        content = ascii_util.raw_string_to_squareized(content, self.res)

        # Embeds characters
        embeddings = ascii_util.squareized_string_to_one_hot(content, self.res)

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
