# Authorship Unknown? A Textual Analysis of the Federalist Papers

##### This analysis seeks to predict the true authors of the Federalist Papers where authorship is currenty contested.  Methods involve scraping the Federalist Papers from the web, consolidating them by authorship, evaluating the frequency of tokens by author, conducting principle components & linear discriminant analysis to generate predictions, and evaluating the confidence of results.  Results suggest with a high degree of confidence that all 11 Federalist Papers with currently contested authorship were in fact writted by James Madison. Below find my full set of code and commentary.

##### The first part of this analysis was written in Python:

```Python3
import requests
from os import getcwd
from os import chdir
from os import listdir as ls
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import numpy
import csv


path = getcwd()
print(path)
chdir('/Users/dylancicero/Desktop/Data_Analysis/Federalist_Papers')
path = getcwd()
print(path)
ls()
```
