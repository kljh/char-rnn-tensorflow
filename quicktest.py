import json
import os, math
import numpy as np

def json_load(filename, asNp = True):
	path = os.path.join("save", 'js', filename)
	with open(path, 'r') as fi:
		data = json.loads(fi.read())
	
	if not asNp:
		return data
	
	
	if len(data)==1 and len(data[0])==1:
		return np.matrix(data[0])
		return np.array(data[0][0])
	elif len(data)==1:
		return np.matrix(data)
		return np.array(data[0])
	else:
		return np.matrix(data)
	
cv= json_load("_chars_vocab.json", False)
embedding = json_load("_iter0__embedding.json") # 3D
inputE = json_load("_iter0_input_embedded.json") # 3D
inputS = json_load("_iter0_input_squeezed.json") # 3D
inputS2 = json_load("_iter1_input_squeezed.json") # 3D
initC = json_load("_iter0_init_state0_c.json") # 2D
initH = json_load("_iter0_init_state0_h.json") # 2D
finalC = json_load("_iter0_final_state0_c.json") # 2D
finalH = json_load("_iter0_final_state0_h.json") # 2D
finalC2 = json_load("_iter1_final_state0_c.json") # 2D
finalH2 = json_load("_iter1_final_state0_h.json") # 2D
var0 = json_load("_iter0_variable0.json")  # 2D
var1 = json_load("_iter0_variable1.json")  # 1D !!

wtfy = json_load("_iter0_input_wtfy.json")

def nprint(x):
	print(x.shape)

def stepprint(x, h, c):
	print()
	print ("x", x)
	print ("h", h)
	print ("c", c)

print("embedding")
nprint(embedding)
print("input")
nprint(inputE)
nprint(inputS)
print("state")
nprint(initC)
nprint(initH)
nprint(finalC)
nprint(finalH)
print("var")
nprint(var0)
nprint(var1)


"""
inputE = np.matrix(inputE[0])
inputS = np.matrix(inputS[0])

initC = np.matrix(initC)
initH = np.matrix(initH)
finalC = np.matrix(finalC)
finalH = np.matrix(finalH)

var0 = np.matrix(var0)
var1 = np.matrix(var1) # np.vector
"""
sigmoid = lambda x: 1. / (1.+math.exp(-x))
sigmoidv = np.vectorize(sigmoid)

dst = lambda a, b : np.linalg.norm(np.subtract(a, b))

n = 24

"""
a = np.array([1,2])
b = np.array([3,4])
#c = np.stack([a, b] , axis=None)
c = np.concatenate((a, b), axis=0)
print(a, b, c)
"""

#https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/rnn_cell_impl.py
def lstm_cell(x, h, c, wtfy, split_permutation):
	
	#  GITHUB 
	#  c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
	#  m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
	#  inputs, m_prev
	v = np.concatenate((x, h), axis=1)
	#print("vconcat")
	#nprint(v)

	v = np.matmul(v, var0)
	v = np.add(v, var1)
	#print("vmultadd")
	#nprint(v)
	
	y = np.negative(c)
	
	splits = [ 
		v[ :, 0:n ],
		v[ :, n:2*n ],
		v[ :, 2*n:3*n ],
		v[ :, 3*n: ] ];
	
	_forget_bias =1.0
	
	#  GITHUB    i = input_gate, j = new_input, f = forget_gate, o = output_gate
	# [ 3, 2, 0, 1 ]  o is last, f is 3rd, i is 1st, ctmp is 2nd  ---- OK
	o = sigmoidv(splits[split_permutation[0]])
	f = sigmoidv(np.add( splits[split_permutation[1]], _forget_bias)) # GITHUB line.857  c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))
	i = sigmoidv(splits[split_permutation[2]])
	ctmp = np.tanh(splits[split_permutation[3]])
	#print("vsplit", o, f, i, ctmp)

	cprev = c
	cnext = np.add( np.multiply(cprev, f),  np.multiply(i, ctmp) )

	hnext = np.multiply(np.tanh(cnext), o)

	return hnext, cnext
	#return cnext[0,0] - finalC[0,0]
	
	print("cnext")
	print(cnext)

	#print("tanh(cnext)", np.tanh(cnext))
	#print("o", o)

	print("finalC")
	print(finalC)
	
	print("hnext") 
	print(hnext)

	print("finalH")

	print(finalH)

	#print(cv["chars"])
	#print(cv["vocab"])

perm = [ 3, 2, 0, 1 ]
#perm = [ 2, 3, 0, 1 ]

h, c = initH, initC

x = inputS
stepprint(x, h, c)
h, c = lstm_cell(x, h, c, wtfy, perm)
print("errc", dst(c, finalC))
print("errh", dst(h, finalH))
#h, c = finalH, finalC

x = inputS2
stepprint(x, h, c)
h, c = lstm_cell(x, h, c, wtfy, perm)
print("errc", dst(c, finalC2))
print("errh", dst(h, finalH2))

print()

def tryall():
	for a in range(4):
		for b in range(4):
			for c in range(4):
				for d in range(4):
					p = [ a, b, c, d ]
					p.sort()
					r = np.add( np.array(p), np.array([ 0, -1, -2, -3 ]) )
					r = np.dot(r, r)
					if r==0:
						h2, c2 = lstm_cell(inputS, initH, initC, wtfy, [ a, b, c, d ])
						err = c2[0,0] - finalC[0,0]
						err = np.linalg.norm(np.subtract(c2, finalC))
						print([ a, b, c, d ], "err", err)

tryall()
