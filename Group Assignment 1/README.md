# Group Assignment 1 | Inverted Index
Implement an inverted index in Python that can be used with Boolean queries with AND operators to retrieve relevant documents

## The Merge of Posting List algorithm
Let say we want to merge two posting lists. Following are the steps to be followed to get merge list:
1. Sort the two posting lists using numeric sort by docID
2. Maintain pointers into both posting lists and walk through them simultaneously, in time linear in the total number of postings entries. 
3. At each step, compare the docID pointed by both pointers. If they are the same, add that docID into the result list, and advance both pointers. Otherwise, advance the pointer pointing to the smaller docID.
4. Repeat step 3 until it reaches the end of one posting list

If the lengths of the postings lists are $m$ and $n$, the merge takes $O(m+n)$ operations. 