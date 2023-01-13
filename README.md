# Ascii Art Dataset

Thank you to the ascii artists which made this dataset possible. 

To start your dataset, follow instructions in `./data_aggregation/README.md`

For use in pytorch, `dataset` implemets a toch Dataset. Optionally, this can be set to serve batches of data that are all of the same size, with different sizes between different batches. This is accomplished by finding all the side dimensions of the ascii art, sorting them, and then putting them in batches.
