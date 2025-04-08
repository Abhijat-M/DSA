
def find3Numbers(A, arr_size, sum):

	
	for i in range( 0, arr_size-2):

		
		for j in range(i + 1, arr_size-1): 
			
			
			for k in range(j + 1, arr_size):
				if A[i] + A[j] + A[k] == sum:
					print("Triplet is", A[i],
						", ", A[j], ", ", A[k])
					return True
	
	return False


A = eval(input("Enter the list of numbers: "))
sum = eval(input("Enter the sum: "))
arr_size = len(A)
find3Numbers(A, arr_size, sum)


