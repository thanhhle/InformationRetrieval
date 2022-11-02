# Group Assignment 3 | Relevance Feedback
Implement an inverted index in Python using PyLucene. A vector space model with cosine similarity can be used with free text query to retrieve top K relevant documents (rank retrieval) from the collection of documents provided. Rocchio's algorithm.

<br/>

## Description
### Indexing
The system accepts a collection from TIME.ALL, process the documents, and construct the index stored in TIME.DIR. A stop list TIME.STP is also used to weed out words. A list of queries from TIME.QUE is also processed, and each query is assigned an unique ID.

<br/>

### Querying
1. The system prompts user to enter the query ID and value of k to retrieve top k documents
2. Top k documents are displayed together with processing time. Precision, Recall, and MAP of the result are calculated
3. The system uses relevance feedback (user relevance feedback, pseudo relevance feedback, or auto relevance feedback) to indicate relevant documents and non-relevant documents to generate a new query with weights using Rocchio algorithm
The system prompts user to enter relevant document IDs and non-relevant document IDs to gnerate a new query with weight using Rocchio algorithm
4. New query is generated with weights displayed
5. User is asked if they want to retrieve top k documents from the new computed query
6. If user enters "y", the new query is processed, and top k documents are displayed together with new Precision, Recall, and MAP values. Otherwise, the system stops

<br/>

## Relevance Feedback
- User Relevance Feedback: The system prompts user to enter relevant document IDs and non-relevant document IDs
- Pseudo Relevance Feedback: The system assumes top 3 documents are relevant and the rest of them are non-relevant
- Auto Relevance Feedback: The system uses the actual relevant documents from TIME.REL to derive relevant documents and non-relevant documents from top k documents retrieved

<br/>

## Requirements and Installations
- Python 3.x versions
- Command-line or Terminal
- Python packages: 
    - NumPy: https://numpy.org/install/
    - PyLucene (Python extension for accessing Java Lucene): https://lucene.apache.org/pylucene/
- JCC: a C++ code generator that makes it possible to call into Java classes from Python via Java's Native Invocation Interface (JNI)

<br/>

## Usage
### Place the following files in a same directory:
- index.py
- time folder:
    - TIME.DIR: place where the PyLucene index is built and storeed as files system locally
    - TIME.ALL: collection - list of documents
    - TIME.REL: list of relevant documents of each query
    - TIME.QUE: list of queries
    - TIME.STP: list of stop words

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

### Execute the index.py file
```
exec(open('index.py').read())
```
<br/>
