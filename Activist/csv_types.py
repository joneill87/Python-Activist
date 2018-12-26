import csv
# We'll need a temporary file-like object, so use a tempfile
from tempfile import TemporaryFile

with TemporaryFile() as t:
    CSVReader = type(csv.reader(t))
    CSVWriter = type(csv.writer(t))
