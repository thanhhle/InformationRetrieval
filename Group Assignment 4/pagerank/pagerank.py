#Python 3.0
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class pagerank:
	
	# function to implement pagerank algorithm
	# input_file - input file that follows the format provided in the assignment description
	def pagerank(self, input_file):
		# record the start time
		start_time = time.time()

		# build the adjacency matrix
		adjacency_matrix = self.build_adjacency_matrix(input_file)

		# build the transition probability matrix
		transition_probability_matrix = self.build_transition_probability_matrix(adjacency_matrix)

		# build the transition probability matrix with teleporting
		teleportation_rate = 0.15
		transition_probability_matrix_with_teleporting = self.build_transition_probability_matrix_with_teleporting(transition_probability_matrix, teleportation_rate)

		# initialize initial probability distribution vector
		probability_vector = np.ones(self.page_count)/self.page_count

		# initialize epsilon value as threshold for indicating steady state
		eps = 1e-5

		# initilize count of iterations
		iterations = 0

		# set limit number of interations
		max_iteration = 200

		# initialize initial power vector
		power_vector = np.ones((self.page_count, max_iteration + 1))/self.page_count

		# calculate the page rank
		while iterations < max_iteration:
			last_probability_vector = probability_vector
			probability_vector = np.dot(last_probability_vector, transition_probability_matrix_with_teleporting)

			for i in range(self.page_count):
				power_vector[i][iterations] = probability_vector[i]

			iterations += 1
			if np.sum(np.abs(probability_vector - last_probability_vector))/self.page_count < eps:
				break

		print("\n\nADJACENCY MATRIX")
		self.print_matrix(adjacency_matrix)

		print("\n\nTRANSITION MATRIX")
		self.print_matrix(transition_probability_matrix)

		print("\n\nTRANSITION MATRIX WITH TELEPORTING")
		self.print_matrix(transition_probability_matrix_with_teleporting)

		# print("\n\nPOWER VECTOR")
		# self.print_matrix(power_vector)

		
		# initialize a dictionary to store the result where key is the page_id and value is the corresponding pagerank value
		result = {k: v for k, v in enumerate(probability_vector)}

		# sort the page rank
		result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

		# record the end time
		end_time = time.time()
		
		# print the output
		print("\n\nNumber of pages: ", self.page_count)
		print("Number of links: ", self.link_count)
		print("\nPageRank calculated in {} iterations ({} seconds)".format(iterations, end_time - start_time))
		print("Page ID\t\tPageRank Value")
		for page_id in result:
			print("{}\t\t{}".format(page_id, result[page_id]))


	# function to build the adjacency matrix
	def build_adjacency_matrix(self, input_file):
		with open(input_file, 'r') as file:  
			lines = file.readlines()

			# record the number of pages 
			self.page_count = (int)(lines[0].strip())

			# record the number of links
			self.link_count = (int)(lines[1].strip())

			# initialize a matrix of 0s with size of page_count x page_count
			adjacency_matrix = np.zeros((self.page_count, self.page_count))
			
			# iterate over each line in the input file to record the link to the matrix
			for line in lines[2:]:
				link = line.split()
				src = int(link[0])
				dst = int(link[1])
				adjacency_matrix[src][dst] = 1

		return adjacency_matrix



	# function to build the transition probability matrix
	# if a row of adjacency matrix A has no 1's, then replace each element by 1/N
	# otherwise, divide each 1 in A by the number of 1's in its row
	def build_transition_probability_matrix(self, adjacency_matrix):
		transition_probability_matrix = np.nan_to_num(np.divide(adjacency_matrix, adjacency_matrix.sum(axis=1)[:, None]), nan=1/len(adjacency_matrix))
		return transition_probability_matrix
			

	# function to build the transition probability matrix with teleporting
	def build_transition_probability_matrix_with_teleporting(self, transition_probability_matrix, teleportation_rate):
		transition_probability_matrix_with_teleporting = transition_probability_matrix * (1 - teleportation_rate)
		transition_probability_matrix_with_teleporting += (teleportation_rate / self.page_count)
		return transition_probability_matrix_with_teleporting


	def print_matrix(self, matrix):
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				print("%6.3f" % matrix[i, j], end="\t")
			print()



def main():
	test1 = os.path.dirname(os.path.abspath(__file__)) + '/test1.txt'
	test2 = os.path.dirname(os.path.abspath(__file__)) + '/test2.txt'
	test3 = os.path.dirname(os.path.abspath(__file__)) + '/test3.txt'
	pagerank().pagerank(test1)
	pagerank().pagerank(test2)
	pagerank().pagerank(test3)


if __name__ == '__main__':
    main()

	
