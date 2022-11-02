#Python 3.0
import os
import collections
import time
import numpy as np
from queue import PriorityQueue

import lucene

from java.io import File
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.util import BytesRefIterator

from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, PostingsEnum, IndexOptions, TermsEnum
from org.apache.lucene.store import FSDirectory

from org.apache.lucene.search import IndexSearcher, MatchAllDocsQuery, DocIdSetIterator
from org.apache.lucene.index import DirectoryReader


class index:

	def __init__(self, path):
		lucene.initVM()

		self.path = os.path.dirname(path)
		self.buildIndex()


	# function to read documents from collection, tokenize and build the index with tokens
	# implement additional functionality to support relevance feedback
	# use unique document integer IDs
	def buildIndex(self):      
		self.index = collections.defaultdict(lambda: collections.defaultdict(list))

        # record the start time
		start_time = time.time()

		# build index using PyLucene
		self.build_lucene_index()

		# construct list of stop words
		with open(self.path + '/TIME.STP', 'r') as lines:
			# read the stop words from the file
			self.stop_list = {word.strip() for word in lines.readlines()}

			# tokenize the stop words
			stream = self.analyzer.tokenStream("", " ".join(self.stop_list))
			stream.reset()

			self.stop_list = []
			while stream.incrementToken():
				stop_word = stream.getAttribute(CharTermAttribute.class_).toString()
				self.stop_list.append(stop_word)

			stream.close()

		# iterate through each document in the lucene index
		self.doc_list = self.get_doc_list()
		for doc in self.doc_list:

			# get the doc_id of the current document
			doc_id = doc.doc

			# get the term vector of the current document
			term_vector = self.reader.getTermVector(doc_id, "content")

			# check if there is term in the document
			if term_vector is not None:
				term_iter = BytesRefIterator.cast_(term_vector.iterator())
				
				# iterate through each term in the document
				while term_iter.next():
					terms_enum = TermsEnum.cast_(term_iter)

					# convert the term from UTF-8 byte format to string
					term = terms_enum.term().utf8ToString()

					# bypass if term is in the stop list
					if term not in self.stop_list:
                        
						# get the postings list of the term
						postings = terms_enum.postings(None, PostingsEnum.ALL)

						# iterate through each posting 
						if postings.nextDoc() is not DocIdSetIterator.NO_MORE_DOCS:

							# get the term frequency
							freq = postings.freq()

							# iterate through each position and record it to the index
							for _ in range(freq):
								pos = postings.nextPosition()
								self.index[term][doc_id].append(pos)


		''' CALCULATE W_TD and IDF_T WEIGHT AND RECORD THEM TO THE INDEX '''
        # iterate through all the terms
		for term in self.index:

			# iterate through all the documents
			for doc_id in self.index[term]:

				# calculate the weighted term frequency of term in doc_id
				tf_td = len(self.index[term][doc_id])
				w_td = 0 if tf_td == 0 else 1 + np.log(tf_td)
				
				# record the weighted term frequency to the index 
				self.index[term][doc_id].insert(0, w_td)
				
			# calculate the inverse document frequency of term
			df_t = len(self.get_docs(term))
			N = len(self.doc_list)
			idf_t = 0 if df_t == 0 else np.log(N/df_t)

			# record the idf weight at index 0 in the corresponding postings list
			self.index[term] = {**{0: [idf_t]}, **self.index[term]}

		# record the end time
		end_time = time.time()
		
		# print the indexing time
		print("\nTF-IDF Index built in", end_time - start_time, "seconds.\n")



	# function to build an index using PyLucene
	def build_lucene_index(self):
		directory_path = self.path + '/TIME.IND'
		collection_path = self.path + '/TIME.ALL'

		# construct the directory to store the index on local file system
		self.directory = FSDirectory.open(File(directory_path).toPath())

		# construct the analyzer which is used to perform analysis on terms appeared in documents to be added in the index
		self.analyzer = StandardAnalyzer()

		# construct and configure an index write
		# always overite existing index to avoid duplicate files
		config = IndexWriterConfig(self.analyzer)
		config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
		self.writer = IndexWriter(self.directory, config)

		# construct reader and searcher
		self.reader = DirectoryReader.open(self.directory)
		self.searcher = IndexSearcher(self.reader)

		# iterate through each line in the collection file to construct doc and add to the IndexWriter
		with open(collection_path, 'r') as file:  
			content = ""
			title = ""
			for line in file.readlines():
				if line.strip():
					if line.startswith("*TEXT"):
						if title != "":
							self.add_doc(title, content)
						content = ""
						title = line[1:9] + ".txt"	
					elif line.startswith("*STOP"):
						self.add_doc(title, content)
					else:
						content += line.strip()

		# close the writer
		self.writer.close()



	# function to add a new document with specific title and content to the IndexWriter
	def add_doc(self, title, content):
		# create a new document
		doc = Document()

		# configure how metadata is stored in the index
		metaType = FieldType()
		metaType.setStored(True)
		metaType.setTokenized(True)
		
		# configure how content data is stored in the index
		# store the doc id, term frequency, and postitions
		contentType = FieldType()
		contentType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
		contentType.setStoreTermVectors(True)
		contentType.setStoreTermVectorPositions(True)				
		contentType.setStored(True)
		contentType.setTokenized(True)

		# add the title and content field to the document
		doc.add(Field("title", title, metaType))
		doc.add(Field("content", content, contentType))

		# add the document to the IndexWriter
		self.writer.addDocument(doc) 



	# function to implement rocchio algorithm
	# pos_feedback - documents deemed to be relevant by the user
	# neg_feedback - documents deemed to be non-relevant by the user
	# return the new query terms and their weights
	def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma):

		# record the start time
		start_time = time.time()

		# construct a variable to store original vector
		q0 = {}

		# consider weight = term frequency in query for free text query
		if isinstance(query_terms, str):
			for query_term in set(self.tokenize(query_terms)):
				q0[query_term] = query_terms.count(query_term)

		# consider the term weight calculate from previous Rocchio for query vector
		else:
			q0 = query_terms

		# get the document vectors of relevant documents
		pos = self.get_doc_vectors(pos_feedback)

		# get the document vectors of non-relevant documents
		neg = self.get_doc_vectors(neg_feedback)

		# generate list of unique terms from original query, positive document vectors, and negative document vectors
		terms = list(q0.keys()) + list(pos.keys()) + list(neg.keys())

		# construct a variable to store new query vector
		qm = collections.defaultdict(float)

		# iterate over each time and compute weight for the new query using Rocchio algorithm
		for term in terms:
			value = alpha * q0.get(term, 0) + beta * 1/len(pos_feedback) * pos.get(term, 0) - gamma * 1/len(neg_feedback) * neg.get(term, 0)
			if value > 0:
				qm[term] = value

		# record the end time
		end_time = time.time()

		# print the result
		print("New query computed in", end_time - start_time, "seconds.")
		print("New query terms with weights:")
		print(dict(qm))
		return qm



	# function to get the doc vectors of a list of docs
	def get_doc_vectors(self, docs):
		doc_vectors = collections.defaultdict(float)

		# iterate through each document
		for doc_id in docs:

            # get the term vector of the current document
			term_vector = self.reader.getTermVector(doc_id, "content")

			# check if there is term in the document
			if term_vector is not None:
				term_iter = BytesRefIterator.cast_(term_vector.iterator())
				
				# iterate through each term in the document
				while term_iter.next():
					# convert the term from UTF-8 byte format to string
					term = TermsEnum.cast_(term_iter).term().utf8ToString()

					# get idf weight of the term
					w_tq = self.get_idf_t(term)

					# get the wtd weight of the term in the document with doc_id
					w_td = self.get_w_td(term, doc_id)
					
					# calculate the tf-idf weight
					tf_idf = w_td * w_tq

					# record the term weight in the document vector
					if tf_idf != 0:
						doc_vectors[term] = tf_idf
		
		return doc_vectors



	# function to find the exact top K retrieval using cosine similarity
	# return at the minimum the document names of the top K documents ordered in decreasing order of similarity score
	def query(self, query_terms, k):
		scores = {}
		
		# record the start time
		start_time = time.time()

		# do not consider weight of each term in the query
		if isinstance(query_terms, str):

			# iterate through each tokenized term in the query
			for query_term in self.tokenize(query_terms):
				
				# calculate the cosine score from the query term to its postings list
				self.cal_cosine_scores(scores, self.get_docs(query_term), query_term)

		# consider the weight new generate query from rocchio
		else:

			# iterate through each term in the query
			for query_term in query_terms:
				
				# calculate the cosine score from the query term to its postings list
				self.cal_cosine_scores(scores, self.get_docs(query_term), query_term, query_terms[query_term])
		
			
		# get top k document with highest score
		result = self.get_top_docs(scores, k)
		
		# record the end time
		end_time = time.time()

		# print the result
		if isinstance(query_terms, str):
			print("\n\nQuery to search:", query_terms)
			print("Number of (top) results:", k)
			print("\nTop {:d} results(s) for the query ' {} ' are:".format(len(result), query_terms))
		else:
			print("\nTop {:d} results(s) for the query are:".format(len(result)))
		print("Doc id, Doc Name, Score")
		for doc in result:
			doc_id = doc[1]
			score = -doc[0]
			title = self.searcher.doc(doc_id).get("title")
			print("{}, {}, {}".format(doc_id, title, score))

		print("\nResults found in", end_time - start_time, "seconds.\n\n")
		return [doc[1] for doc in result]
		


	# function to print the terms and posting lists in the index
	def print_dict(self):
		print("\nPrint the terms and posting lists")
		output = collections.defaultdict(lambda: collections.defaultdict(list))
		for term in self.index:
			postings = self.get_docs(term)
			for posting in postings:    
				output[term][posting[0]] = posting[1][1:]

			print(term, list(output[term].items()))
		

		
	# function to print the documents and their document id
	def print_doc_list(self):
		for doc in self.doc_list:
			doc_id = doc.doc
			title = self.searcher.doc(doc_id).get("title")
			print("Doc ID:", doc_id, "==>", title)
		


	# function to retrieve all the documents
	def get_doc_list(self):
		# construct a special query that match all docucments
		query = MatchAllDocsQuery()

		# retrieve all possible documents that match the special query
		docs = self.searcher.search(query, self.searcher.count(query))
		return docs.scoreDocs
    

    
    # function to get the log frequency weight of term in document with doc_id
	def get_w_td(self, term, doc_id):
		w_td = 0
		try:
			w_td = self.index[term][doc_id][0]
		except:
			w_td = 0
			
		return w_td
     

    
    # function to get the inverse document frequency of term
	def get_idf_t(self, term):
		idf_t = 0
		try:
			idf_t = self.index[term][0][0]
		except:
			idf_t = 0
			
		return idf_t

	

	# function to get list of doc ids that the input term appears
	def get_docs(self, term):
		return [doc_id for doc_id in self.index[term]][1:]
		


	# function to get the document length
	def get_doc_length(self, doc_id):
		content = self.searcher.doc(doc_id).get("content")
		tokens = self.tokenize(content)
		return len(tokens)



	# function to get the top k retrievals
	def get_top_docs(self, scores, k):
		result = []
			
		# iterate through each document
		queue = PriorityQueue()
		for doc_id in scores:
			
			# normalize the score by document length
			scores[doc_id] /= self.get_doc_length(doc_id)
				
			# add the score to a priority queue in negative number so that it can be retrieved descendingly
			queue.put((scores[doc_id] * -1, doc_id))

		# get top k value from the queue
		while not queue.empty() and len(result) < k:
			result.append(queue.get())
		
		return result



	# function to calculate the cosine scores from the query term to each of documents in the input list
	def cal_cosine_scores(self, scores, docs, query_term, term_weight=None):

		# get idf weight of the term
		w_tq = self.get_idf_t(query_term) if term_weight == None else term_weight
		
		# iterate through each document
		for doc_id in docs:
			
			# get the wtd weight of the term in the document with doc_id
			w_td = self.get_w_td(query_term, doc_id)

			# calculate the tf-idf weight
			tf_idf = w_td * w_tq
			
			# accumulate the score
			scores[doc_id] = scores.get(doc_id, 0) + tf_idf

	
	
	# convert an input string into a list of tokenized terms and bypass stop words
	def tokenize(self, string):
		stream = self.analyzer.tokenStream("", string)
		stream.reset()

		tokens = []
		while stream.incrementToken():
			token = stream.getAttribute(CharTermAttribute.class_).toString()
			if token not in self.stop_list:
				tokens.append(token)

		stream.close()
		return tokens



def main():
	collection_path = os.path.dirname(os.path.abspath(__file__)) + '/time/TIME.ALL'
	a = index(collection_path)

	queries = get_queries()
	for query_id in queries:
		print(query_id, "-", queries[query_id])

	while True:
		print("\n=========================================================================")
		query_id = int(input("\nEnter query id (1 - " + str(len(queries)) + "): "))

		query_terms = queries[query_id]
		relevant_docs = get_relevant_docs()[query_id]

		print("Query:", query_terms)
		print("Relevant docs:", relevant_docs)

		k = int(input("\nEnter value of k: "))
	
		result = a.query(query_terms, k)
		cal_precision_recall_map(result, relevant_docs)
	
		alpha = 1
		beta = 0.75
		gamma = 0.15

		i = 1
		while True:
			print("\n\n=== Rocchio Algorithm ===")
			print("\nIteration:", i)

			pos_feedback, neg_feedback = user_relevance_feedback()
			# pos_feedback, neg_feedback = pseudo_relevance_feedback(result)
			# pos_feedback, neg_feedback = auto_relevance_feedback(result, relevant_docs)

			query_terms = a.rocchio(query_terms, pos_feedback, neg_feedback, alpha, beta, gamma)

			answer = input("\nContinue with new query (y/n): ")
			if answer.lower() == "y":
				result = a.query(query_terms, k)
				cal_precision_recall_map(result, relevant_docs)
				i += 1
			else:
				break

	
	# a.print_doc_list()
	# a.print_dict()   



# function to get list of queries
def get_queries():
	query_path = os.path.dirname(os.path.abspath(__file__)) + '/time/TIME.QUE'
	queries = {}

	with open(query_path, 'r') as file:  
		query = ""
		for line in file.readlines():
			if line.strip():
				if line.startswith("*FIND") or line.startswith("*STOP"):
					if query != "":
						queries[len(queries) + 1] = query
					query = ""
				else:
					query += line.strip()
	
	return queries



# function to get list of relevant docs of each query
def get_relevant_docs():
	relevant_docs_path = os.path.dirname(os.path.abspath(__file__)) + '/time/TIME.REL'
	relevant_docs = {}

	with open(relevant_docs_path, 'r') as file:  
		for line in file.readlines():
			if line.strip():
				line = line.strip().split()
				query_id = int(line[0])
				docs = [int(doc_id) - 1 for doc_id in line[1:]]
				relevant_docs[query_id] = docs

	return relevant_docs



# function to calculate precision, recall, and MAP
def cal_precision_recall_map(result, relevant_docs):
	relevant_retrieved_docs = []
	map = 0

	for i, doc_id in enumerate(result):
		if doc_id in relevant_docs:
			relevant_retrieved_docs.append(doc_id)
			map += len(relevant_retrieved_docs)/(i + 1)

	precision = len(relevant_retrieved_docs)/len(result)
	recall = len(relevant_retrieved_docs)/len(relevant_docs)
	map = map/len(relevant_docs)

	print("Relevant docs that are retrieved:", relevant_retrieved_docs)
	print("Precision:", precision)
	print("Recall:", recall)
	print("MAP:", map)



# function to get user relevance feedback
def user_relevance_feedback():
	print("\n--- User Relevance Feedback ---")
	pos_feedback = input("Enter relevant document ids separated by space: ")
	neg_feedback = input("Enter non relevant document ids separated by space: ")

	pos_feedback = list(map(int, pos_feedback.split(" ")))
	neg_feedback = list(map(int, neg_feedback.split(" ")))

	return pos_feedback, neg_feedback



# function to derive relevant and non-relevant documents from the result
def pseudo_relevance_feedback(result):
	print("\n--- Pseudo Relevance Feedback ---")

	# assume top 3 documents are relevant and the rest are non-relevant
	pos_feedback = result[:3]
	neg_feedback = result[3:]

	# print the result
	print("Relevant document ids:", pos_feedback)
	print("Non relevant document ids:", neg_feedback)

	return pos_feedback, neg_feedback



# function to derive relevant and non-relevant documents using known actual relevant documents
def auto_relevance_feedback(result, relevant_docs):
	print("\n--- Auto Relevance Feedback ---")
	pos_feedback = list(set(result) & set(relevant_docs))
	neg_feedback = list(set(result) - set(pos_feedback))

	# print the result
	print("Relevant document ids:", pos_feedback)
	print("Non relevant document ids:", neg_feedback)

	return pos_feedback, neg_feedback




if __name__ == '__main__':
    main()
    