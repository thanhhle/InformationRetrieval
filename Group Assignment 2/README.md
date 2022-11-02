# Group Assignment 2 | Inverted Index
Implement a vector space model with cosine similarity in Python that can be used with free text query to retrieve top K relevant documents (rank retrieval)
<br/><br/>

## Description
The system accepts a collection, process the documents, and construct the index. A stop list is also used to weed out words.

The rank retrieval methods supported include: 
1. Exact Retrieval: ranking documents based on TF-IDF scores
2. Inexact Retrieval using Champion List: ranking documents using pre-computed champion lists built on the weighted term frequency WTD
3. Inexact Retrieval using Index Elimination: ranking documents using only half the queries terms sorted in decreasing order of their IDF values
4. Inexact Retrieval using Cluster Pruning: ranking documents using pre-computed cluster pruning (randomly pick √N leaders and use them to implement the cluster pruning)
<br/><br/>

## Requirements and Installations
- Python 3.x versions
- Command-line or Terminal
- Addition Python packages: NumPy (https://numpy.org/install/)
<br/><br/>


## Usage
### Place the following files in a same directory:
- index.py
- collection: folder contains text files of documents to be indexed
- stop-list.txt: text files contains list of stop words to be ommited
<br/><br/>

### Navigate the working directory to that directory:
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

### Build the index
```
a = index([collection path])
```
or
```
a = index(os.path.join(os.getcwd(), 'collection'))
```
<br/>

### Run the queries
#### Exact Retrieval: exact_query
```
a.exact_query([query terms], [number of documents to be retrieved])
```
Ex: Using Exact retrieval for top 5 documents of: 
```
a.exact_query('government party political', 5)
```
<br/>
  
#### Inexact Retrieval using Champion List: inexact_query_champion
```
a.inexact_query_champion([query terms], [number of documents to be retrieved])
```
Ex: Using Inexact retrieval using Champion List for top 7 documents of: world, war, president, politician
```
a.inexact_query_champion('world war president politician', 7)
```
<br/>
  
#### Inexact Retrieval using Index Elimination: inexact_query_index_elimination
```
a.inexact_query_index_elimination([query terms], [number of documents to be retrieved])
```
Ex: Using Inexact retrieval using Index Elimination for top 3 documents of: university, student
```
a.inexact_query_index_elimination('university student', 3)
```
<br/>

#### Inexact Retrieval using Cluster Pruning: inexact_query_cluster_pruning
```
a.inexact_query_cluster_pruning('[your query terms]', [number of documents to be retrieved])
```
Ex: Using Inexact retrieval using Cluster Pruning for top 10 documents of: military, conference, leader, citizen, soldier
```
a.inexact_query_cluster_pruning('military conference leader citizen soldier', 10)
```