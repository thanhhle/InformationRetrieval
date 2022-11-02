import re
import os
import collections
import time

class index:
    def __init__(self, path):
        os.chdir(path)
        self.buildIndex()



    # function to read documents from collection, tokenize and build the index with tokens
    # index should also contain positional information of the terms in the document --- term: [idft, (ID1, wt1d1, [pos1,pos2,..]), (ID2, wt1d2, [pos1,pos2,…]),….]
    # use unique document IDs
    def buildIndex(self):
        self.doc_list = {}
        self.index = collections.defaultdict(lambda: collections.defaultdict(list))
        
        # record the start time
        start_time = time.time()

        # iterate through all documents in the collection
        for filename in os.listdir():
            # check if the document is in text format
            if filename.endswith('.txt'):
                doc_id = len(self.doc_list)
                
                # add the document into a list
                self.doc_list[doc_id] = filename
                
                # tokenize the document and build the index with corresponding doc ID and positional information
                for pos, term in enumerate(self.tokenize(filename)):
                    self.index[term][doc_id].append(pos)
                  
        # sort the index
        self.index = dict(sorted(self.index.items()))
        
        # record the end time
        end_time = time.time()
        
        print("Index built in", end_time - start_time, "seconds.\n")



    # function to identify relevant docs using the index
    def and_query(self, query_terms):
        # record the start time
        start_time = time.time()
        
        # set the posting lists of the first term in the query as the result lists
        result = [doc_id for doc_id in self.index[query_terms[0]]]
        
        # iterate through all other terms in the query
        for query_term in query_terms[1:]:
            # get the posting lists of the current term
            docs = [doc_id for doc_id in self.index[query_term]]
            
            # merge the result lists with the posting lists of the current term
            result = self.and_merge(result, docs)
        
        # record the end time
        end_time = time.time()
        
        print("Results for the Query:", " AND ".join(query_terms))
        print("Total Docs retrieved:", len(result))
        
        for doc_id in result:
            print(self.doc_list[doc_id])
            
        print("Retrieved in", end_time - start_time, "seconds.\n")
	
    

    # function to print the terms and posting lists in the index
    def print_dict(self):
        print("\nPrint the terms and posting lists")
        for term in self.index:
            print(term, list(self.index[term].items()))
        
        

    # function to print the documents and their document id
    def print_doc_list(self):
        print("\nPrint the documents and their Doc ID")
        for doc_id in self.doc_list:
            print("Doc ID:", doc_id, "==>", self.doc_list[doc_id])
        
        
        
    # function to tokenize a document and return a list of terms
    def tokenize(self, filename):
        doc = open(filename).read()
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
    


def main():
    a = index('/Users/thanhle/Documents/CPP/Courses/Fall 2022/CS 5180/Group Assignment/collection/')
    '''
    a.and_query(['with', 'without', 'yemen'])
    a.and_query(['with', 'without', 'yemen', 'yemeni'])
    a.and_query(['ready', 'for', 'right'])
    a.and_query(['girl', 'time', 'hard', 'after'])
    a.and_query(['down', 'like', 'fire'])
    a.and_query(['pretty', 'never', 'know'])
    a.and_query(['world', 'war', 'president', 'government'])
    '''
               
    a.print_doc_list()
    # a.print_dict()   

    

if __name__ == '__main__':
    main()



    
    
