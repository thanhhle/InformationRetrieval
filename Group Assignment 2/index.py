import re
import os
import collections
import time
import numpy as np
import random
from queue import PriorityQueue

    
class index:
    
    def __init__(self, path):
        self.path = path
        with open(os.path.dirname(self.path) + '/stop-list.txt', 'r') as lines:  
            self.stop_list = {word.strip() for word in lines.readlines()}
        
        self.buildIndex()


    # function to read documents from collection, tokenize and build the index with tokens
    # index should also contain positional information of the terms in the document --- term: [idf(ID1, w1, [pos1,pos2,..]), (ID2, w2, [pos1,pos2,…]),….]
    # use unique document IDs
    def buildIndex(self):
        self.doc_list = {}
        self.index = collections.defaultdict(lambda: collections.defaultdict(list))
        
        # record the start time
        start_time = time.time()

        ''' GENERATE POSTINGS LIST WITH POSITION INFORMATION '''
        # iterate through all documents in the collection
        for doc in os.listdir(self.path):
            # check if the document is in text format
            if doc.endswith('.txt'):
                # the doc_id of the first document in the collection is 1, by convention
                doc_id = len(self.doc_list) + 1
                
                # add the document into a list
                self.doc_list[doc_id] = doc
                
                # tokenize the document and build the index with corresponding doc ID and positional information
                # the first position in a document is 1
                for pos, term in enumerate(self.tokenize(doc)):
                    
                    # bypass if term is in the stop list
                    if term not in self.stop_list:
                        
                        # record the position to the posting list
                        self.index[term][doc_id].append(pos + 1)
            
        # sort the index
        self.index = dict(sorted(self.index.items()))
            
        
        ''' CALCULATE W_TD and IDF_T WEIGHT AND RECORD THEM TO THE INDEX '''
        # iterate through all the terms
        for term in self.index:

            # iterate through all the documents
            for doc_id in self.index[term]:

                # calculate the weighted term frequency of term in doc_id
                tf_td = len(self.index[term][doc_id])
                w_td = 0 if tf_td == 0 else 1 + np.log10(tf_td)
                
                # record the weighted term frequency to the index 
                self.index[term][doc_id].insert(0, w_td)
                
            # calculate the inverse document frequency of term
            df_t = len(self.get_docs(term))
            N = len(self.doc_list)
            idf_t = 0 if df_t == 0 else np.log10(N/df_t)

            # record the idf weight at index 0 in the corresponding postings list
            self.index[term] = {**{0: [idf_t]}, **self.index[term]}
            
            
        ''' BUILD THE CHAMPION LIST '''
        self.build_champion_list()
        
        
        ''' BUILD THE CLUSTER PRUNING '''
        self.build_cluster_pruning()
        
        
        # record the end time
        end_time = time.time()
        print("Index built in", end_time - start_time, "seconds.\n")



    # function to build the champion list
    def build_champion_list(self):
        self.champion_list = collections.defaultdict(list)
        queue = PriorityQueue() 
        
        # iterate through all the terms
        for term in self.index:

            # iterate through all the documents
            for doc_id in self.get_docs(term):
                
                # get the weighted term frequency in the document
                w_td = self.get_w_td(term, doc_id)
                
                # add the weight to a priority queue together with the corresponding doc_id
                queue.put((w_td * -1, doc_id))
            
            # get the idf weight of the term
            idf_t = self.get_idf_t(term)
            
            # determine number of documents r in champion list (rarer terms should have bigger champions list)
            r = 5 + idf_t * len(self.doc_list)/4
            
            # generate the champion list for each term
            while not queue.empty() and len(self.champion_list[term]) < r:
                self.champion_list[term].append(queue.get()[1])
    
    

    # function to build the cluster pruning
    def build_cluster_pruning(self):
        self.cluster_pruning = collections.defaultdict(list)
        
        # pick N docs at random to be leaders
        self.leader_list = []
        for _ in range(int(np.sqrt(len(self.doc_list)))):
            self.leader_list.append(random.randint(1, len(self.doc_list)))

        # consider other docs as followers
        follower_list = list(self.doc_list.keys() - self.leader_list)

        # iterate through the followers
        for doc_id in follower_list:
            
            # get the terms from each follower
            doc_terms = set(self.tokenize(self.doc_list[doc_id]))
            
            # bypass terms are in the stop list
            doc_terms -= self.stop_list
            
            if len(doc_terms) > 0:
                
                scores = {}
                
                # iterate through each term in the follower
                for doc_term in doc_terms:
                    
                    # calculate the cosine score from the term to its leaders
                    self.cal_cosine_scores(scores, self.leader_list, doc_term)
                    
                # get top 1 document with highest score
                result = self.get_top_docs(scores, 1)
                
                # set this top document as the nearest leader
                nearest_leader = result[0][1]
        
                # record the nearest leader for each follower
                self.cluster_pruning[nearest_leader].append(doc_id)
            
                 
            
    # function to identify relevant docs using the index
    def and_query(self, query_terms):
        result = []
        
        # record the start time
        start_time = time.time()
        
        # bypass the terms that are stop words
        query_terms = list(set(query_terms) - self.stop_list)
        
        # iterate through all other terms in the query     
        for i in range(len(query_terms)):
            query_term = query_terms[i]
            
            # check if the query term is in the index
            if query_term in self.index:
                if i == 0:
                    # set the posting lists of the first term in the query as the result lists
                    result = self.get_docs(query_term)
                    
                else:
                    # get the posting lists of the current term
                    docs = self.get_docs(query_term)
                    
                    # merge the result lists with the posting lists of the current term
                    result = self.and_merge(result, docs)
            else:
                result = []
                break
        
        # record the end time
        end_time = time.time()
        
        # print the result
        print("Results for the Query:", " AND ".join(query_terms))        
        self.print_retrieved_docs(result, end_time - start_time)
        
        
    
    # function to exact top K retrieval (method 1)
	# return at the minimum the document names of the top K documents ordered in decreasing order of similarity score
    def exact_query(self, query_terms, k):
        scores = {}
        
        # record the start time
        start_time = time.time()
        
        # convert the free text query to a list of tokenized terms and bypass stop words 
        query_terms = self.process_query(query_terms)
        
        # iterate through each term in the query
        for query_term in query_terms:
            
            # calculate the cosine score from the query term to its postings list
            self.cal_cosine_scores(scores, self.get_docs(query_term), query_term)
            
        # get top k document with highest score
        result = self.get_top_docs(scores, k)
        
        # record the end time
        end_time = time.time()
        
        # print the result
        print("Results for the Exact Top {k} Retrievals of:".format(k = k), ", ".join(query_terms))
        self.print_retrieved_docs(result, end_time - start_time)
	
    
    
    # function to inexact top K retrieval using champion list (method 2)
	# return at the minimum the document names of the top K documents ordered in decreasing order of similarity score
    def inexact_query_champion(self, query_terms, k):
        scores = {}
        
        # record the start time
        start_time = time.time()
        
        # convert the free text query to a list of tokenized terms and bypass stop words 
        query_terms = self.process_query(query_terms)
        
        # iterate through each term in the query
        for query_term in query_terms:
            
            # calculate the cosine score from the query term to its champion list
            self.cal_cosine_scores(scores, self.champion_list[query_term], query_term)
            
        # get top k document with highest score
        result = self.get_top_docs(scores, k)
        
        # record the end time
        end_time = time.time()
        
        # print the result
        print("Results for the Inxact Top {k} Retrievals using Champion List of:".format(k = k), ", ".join(query_terms))
        self.print_retrieved_docs(result, end_time - start_time)
        
        
    
    # function to inexact top K retrieval using index elimination (method 3)
	# return at the minimum the document names of the top K documents ordered in decreasing order of similarity score
    def inexact_query_index_elimination(self, query_terms, k):
        high_idf_terms = []
        queue = PriorityQueue()
        
        # record the start time
        start_time = time.time()
        
        # convert the free text query to a list of tokenized terms and bypass stop words 
        query_terms = self.process_query(query_terms)
        
        # iterate through all terms in the query
        for query_term in query_terms:
            
            # calculate the idf value of the term
            idf_t = self.get_idf_t(query_term)
            
            # add the negative idf values to a priority queue to later sort in decreasing order
            queue.put((idf_t * -1, query_term))
        
        # pick half of the query terms with highest idf values
        while not queue.empty() and len(high_idf_terms) < len(query_terms)/2:
            high_idf_terms.append(queue.get()[1])
        
        # retrieve top k documents using these query values
        scores = {}
        
        # iterate through each term in the set of highest idf values
        for query_term in high_idf_terms:
            
            # calculate the cosine score from the query term to its posting list
            self.cal_cosine_scores(scores, self.get_docs(query_term), query_term)
            
        # get top k document with highest score
        result = self.get_top_docs(scores, k)
        
        # record the end time
        end_time = time.time()
        
        # print the result
        print("Results for the Inxact Top {k} Retrievals using Index Elimination of:".format(k = k), ", ".join(query_terms))
        self.print_retrieved_docs(result, end_time - start_time)


    
    # function to inexact top K retrieval using cluster pruning (method 4)
	# return at the minimum the document names of the top K documents ordered in decreasing order of similarity score
    def inexact_query_cluster_pruning(self, query_terms, k):
        leader_scores = {}
        scores = {}
        
        # record the start time
        start_time = time.time()
        
        # convert the free text query to a list of tokenized terms and bypass stop words 
        query_terms = self.process_query(query_terms)
        
        # iterate through each term in the query
        for query_term in query_terms:
            
            # calculate the cosine score from the query term to the leaders
            self.cal_cosine_scores(leader_scores, self.leader_list, query_term)
            
        # get the leaders ordered in decreasing order of similarity score
        leader_result = self.get_top_docs(leader_scores, len(self.leader_list))
        
        i = 0        
        result = []
        
        # look for the next best leader if the current one does not get enough top k results
        while len(result) < k and i < len(leader_result):
  
            # get the best leader at the moment
            leader = leader_result[i][1]
            
            # iterate through each term in the query
            for query_term in query_terms:
                
                # construct a candidate set which consists of leader together with its followers
                candidates = self.cluster_pruning[leader] + [leader]

                # calculate the cosine score from the query term to the candidates
                self.cal_cosine_scores(scores, candidates, query_term)
                
            # get top k documents with highest score
            result += self.get_top_docs(scores, k - len(result))
            i += 1
        
        
        # record the end time
        end_time = time.time()
        
        print("Results for the Inxact Top {k} Retrievals using Cluster Pruning of:".format(k = k), ", ".join(query_terms))
        self.print_retrieved_docs(result, end_time - start_time)
        
        
    
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
        return self.index[term][0][0]


    
    # function to get list of doc ids that the input term appears
    def get_docs(self, term):
        return [doc_id for doc_id in self.index[term]][1:]
        
    
    
    # function to get the document length
    def get_doc_length(self, doc_id):
        doc = self.doc_list[doc_id]
        return len(self.tokenize(doc))
    
    
    
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
    def cal_cosine_scores(self, scores, docs, query_term):

        # get idf weight of the term
        wtq = self.get_idf_t(query_term)
        
        # iterate through each document
        for doc_id in docs:
            
            # get the wtd weight of the term in the document with doc_id
            wtd = self.get_w_td(query_term, doc_id)
            
            # calculate the if-idf weight
            tf_idf = wtd * wtq
            
            # accumulate the score
            scores[doc_id] = scores.get(doc_id, 0) + tf_idf
    
        
        
    # function to tokenize a document and return a list of terms
    def tokenize(self, filename):
        doc = open(self.path + "/" + filename).read()
        return re.findall("[\w]+", doc.lower())
        
        
        
    # function to perform "AND merge" on two sorted lists
    def and_merge(self, list1, list2):
        result = []
            
        # construct the pointers
        i, j = 0, 0
            
        # move the pointers over all entries in the two lists simultaneously until it reaches the end of either one of them
        while i < len(list1) and j < len(list2):
            # if two values being pointed are equal, add the value to the result array and move both pointers
            if list1[i] == list2[j]:
                result.append(list1[i])
                i += 1
                j += 1
                    
            # if the value in list1 is smaller than list2, advance the pointer of list1
            elif list1[i] < list2[j]:
                i += 1
                    
            # if the value in list1 is greater than list2, advance the pointer of list2
            else:
                j += 1
            
        return result



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
        print("\nPrint the documents and their Doc ID")
        for doc_id in self.doc_list:
            print("Doc ID:", doc_id, "==>", self.doc_list[doc_id])
            
            
            
    # function to print the top retrieved documents with their corresponding cosine scores
    def print_retrieved_docs(self, result, time):
        print("Total Docs retrieved:", len(result))
        for doc_id in result:
            if type(doc_id) != int:
                print("Doc: %-*s Score: %s" % (20, self.doc_list[doc_id[1]], -doc_id[0]))
            else:
                print("Doc: %-*s" % (20, self.doc_list[doc_id]))
            
        print("Retrieved in", time, "seconds.\n\n")



    # convert the free text query to a list of tokenized terms and bypass stop words
    def process_query(self, query_terms):
        # convert the free text query to a list of tokenized terms
        terms = re.split(r"\W+", query_terms.lower())

        # bypass terms are in the stop list
        return list(set(terms) - self.stop_list)


    
    def cal_w_tq(self, query_term, query_terms):
        qf_t = len(self.get_docs(term))

        N = len(self.doc_list)
        idf_t = 0 if df_t == 0 else np.log10(N/df_t)
        




def main():
    a = index(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collection'))


    '''
    a.and_query(['with', 'without', 'yemen'])
    a.and_query(['with', 'without', 'yemen', 'yemeni'])
    a.and_query(['ready', 'for', 'right'])
    a.and_query(['girl', 'time', 'hard', 'after'])
    a.and_query(['down', 'like', 'fire'])
    a.and_query(['pretty', 'never', 'know'])
    a.and_query(['world', 'war', 'president', 'government'])
    '''
    
    
    queries = [ 
                'government party political',
                'world war president politician',
                'university student',
                'military conference leader citizen soldier',
                'plane fire fight',
              ]


    for query in queries:
        a.exact_query(query, 10)
        a.inexact_query_champion(query, 10)
        a.inexact_query_index_elimination(query, 10)
        a.inexact_query_cluster_pruning(query, 10)
        print("------------------------------------------------------------------------------------------------------")
   
    
    # a.print_doc_list()
    # a.print_dict()   

    

if __name__ == '__main__':
    main()
    
    
