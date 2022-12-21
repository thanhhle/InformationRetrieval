# Group Assignment 4 | Page Rank Algorithm - Distributed Indexing
- Implement the iterative algorithm that uses power method to generate the page rank.
- Implement the distributed index that utilizes map reduce paradigm and Hadoop cluster to index documents.

<br/>

## Description
### Page Rank
The system reads a directed graph of the webpages from a text file with the following format:
```
(# of pages)
(# of links)
(src page1)(dst page1) # an outgoing link from webpage page1 to page2
(src page2)(dst page2)
...
```
The adjancy matrix, transition probability matrix, and transition probability matrix with teleporting are generated. The probability distribution vector is then calculated using iterative algorithm with the power method to find the steady state when the epsilon value or number of maximum iteration is reached. The program prints out the matrices and page rank for the pages (both page id and pagerank value). 

<br/>

### Distributed Index
The system indexes a dataset of tweets (tweets.xlsx) using both traditional indexing and distributed indexing that uses map reduce along with Hadoop cluster. The runtime of both methods is compared to analyze the speedup of using map reduce.

<br/>

## Requirements and Installations
- Python 3.x versions
- Command-line or Terminal
- Python packages: 
    - NumPy: https://numpy.org/install/
    - PyLucene (Python extension for accessing Java Lucene): https://lucene.apache.org/pylucene/
- JCC: a C++ code generator that makes it possible to call into Java classes from Python via Java's Native Invocation Interface (JNI)
- mrjob: a python library for MapReduce developed by YELP that allows writing MapReduce code using a Python running on Hadoop (https://mrjob.readthedocs.io/en/latest/)
- Apache Hadoop: a framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models (https://hadoop.apache.org)

<br/>

## Usage
### Place the python and input files in a same directory:
- pagerank: pagerank.py and test.txt files
- mapreduce: index.py and tweets.xlsx files

<br/>

### Navigate the system to the directory
```
cd [directory path]
```
<br/>

### Start a Python interactive session
```
python3
```
or
```
python
```
<br/>

### Execute the python file
```
exec(open('pagerank.py').read())
```
```
exec(open('index.py').read())
```
<br/>
