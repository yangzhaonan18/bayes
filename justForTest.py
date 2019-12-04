# index = 19
# W_num = 19

# i = index // W_num
# j = index % W_num

# print(i , j)

# index = W_num * i + j
# print(index)





def ij2index(i, j):
    W_num =19
    return W_num * i + j

def index2ij(index_cell):
    W_num = 19
    return  index_cell // W_num, index_cell % W_num

def find_7index(index):
	H_num = 32
	W_num = 19
	max_inde = 607  # have 608 cell
	if index > 607 or index < 0:
		print("Input index error !!!")
	ij_7s = [[-1, -1] for i in range(7)]
	i, j = index2ij(index)

	# ##########
	# if index out of range,  set -1
	# ########
	ij_7s[0][0], ij_7s[0][1] = i, j
	ij_7s[1][0], ij_7s[1][1] =  i + 2 if i + 1 <= H_num - 1 else -1, j  
	ij_7s[2][0], ij_7s[2][1] =  i + 1 if i + 1 <= H_num - 1 else -1, j + 1 if j + 1 <= W_num -1 else -1
	ij_7s[3][0], ij_7s[3][1] =  i - 1 if i - 1 >= 0 else -1, j + 1 if j + 1 <= W_num -1 else -1
	ij_7s[4][0], ij_7s[4][1] =  i - 2 if i - 1 >= 0 else -1, j
	ij_7s[5][0], ij_7s[5][1] =  i - 1 if i - 1 >= 0 else -1, j
	ij_7s[6][0], ij_7s[6][1] =  i + 1 if i + 1 <= H_num -1 else -1, j
	# print("input index = ", index)
	# print("current i j index is = ", i, j)
	# print("output 7 index = ", ij_7s)
	for i in range(len(ij_7s)):
		index_7s[i] = ij2index(ij_7s[i][0], ij_7s[i][1])
	return ij_7s, index_7s
 
 
	 



if __name__ == "__main__":
 	find_7index(18)
 	a = 1
 	b = 10 if a > 3 else 100 
 	print(b)
 	print(19 % 19)
