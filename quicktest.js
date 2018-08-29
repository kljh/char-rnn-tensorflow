
const fs = require('fs');
const data = JSON.parse(fs.readFileSync("save/js/model_data.json"));

function check_dimensions(data) {
	if (data.length==1 && data[0].length==1)
		return data[0][0];
	if (data.length==1)
		return data[0];
	else
		return data;
}

embedding = check_dimensions(data["embedding"])
var0 = check_dimensions(data["cell_variables"][0])
var1 = check_dimensions(data["cell_variables"][1])
softmax_w = check_dimensions(data["softmax_w"])
softmax_b = check_dimensions(data["softmax_b"])

inputE = check_dimensions(data["iterations"][0]["input_embedded"])
inputS = check_dimensions(data["iterations"][0]["input_squeezed"])
inputS2 = check_dimensions(data["iterations"][1]["input_squeezed"])
initC = check_dimensions(data["iterations"][0]["init_state_c"])
initH = check_dimensions(data["iterations"][0]["init_state_h"])
finalC = check_dimensions(data["iterations"][0]["final_state_c"])
finalH = check_dimensions(data["iterations"][0]["final_state_h"])
finalC2 = check_dimensions(data["iterations"][1]["final_state_c"])
finalH2 = check_dimensions(data["iterations"][1]["final_state_h"])

function shape(a) {
	if (a.length==0 || !Array.isArray(a[0])) {
		return [a.length];
	} else {
		var tmp = shape(a[0]);
		tmp.unshift(a.length);
		return tmp;
	}
}

function nprint(x) {
	console.log(shape(x));
}

function stepprint(x, h, c) {
	console.log()
	console.log ("x", x)
	console.log ("h", shape(h))
	console.log ("c", shape(c))
}

console.log("embedding")
nprint(embedding)
console.log("input")
nprint(inputE)
nprint(inputS)
console.log("state")
nprint(initC)
nprint(initH)
nprint(finalC)
nprint(finalH)
console.log("var")
nprint(var0)
nprint(var1)


// https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/rnn_cell_impl.py
function lstm_cell(x, h, c) {
	var n = x.length;
	
	/* GITHUB 
	#  c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
	#  m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
	#  inputs, m_prev
	*/
	var v = [].concat(x, h)
	console.log("vconcat")
	nprint(v)

	var w = vecmatmul(v, var0)
	var wb = vadd(w, var1)
	console.log("vmultadd")
	nprint(wb)
	
	//  GITHUB    i = input_gate, j = new_input, f = forget_gate, o = output_gate
	var wbi = wb.slice(0,n),
		wbc = wb.slice(n,2*n),
		wbf = wb.slice(2*n, 3*n),
		wbo = wb.slice(3*n);
	
	//console.log("vsplit", shape(wb), wbo, wbf, wbi, wbc)

	var _forget_bias =1.0
	
	var o = sigmoidv(wbo),
		f = sigmoidv(vaddcste( wbf, _forget_bias)), // GITHUB line.857  c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))
		i = sigmoidv(wbi),
		ctmp = tanhv(wbc);
		
	//console.log("vsplit", o, f, i, ctmp)

	cprev = c
	cnext = vadd( vmul(cprev, f),  vmul(i, ctmp) )

	hnext = vmul(tanhv(cnext), o)

	return { h: hnext, c: cnext };
}

function test() {
	var [ h, c ] = [ initH, initC ]

	x = inputS
	stepprint(x, h, c)
	var { h, c } = lstm_cell(x, h, c);
	console.log("errc", dst(c, finalC))
	console.log("errh", dst(h, finalH))
	//h, c = finalH, finalC

	x = inputS2
	stepprint(x, h, c)
	var { h, c } = lstm_cell(x, h, c) ;
	console.log("errc", dst(c, finalC2))
	console.log("errh", dst(h, finalH2))
}


// defining few NumPy utility fucntions 

function vecmatmul(vec, mat) {
	var nv = vec.length,  
		n = mat.length, 
		m = mat[0].length;
	if (nv!=n) throw new Error("vecmatmult dimension mismatch");

	var w = new Array(m);
	for (var j=0; j<m; j++) {
		var tmp = 0;
		for (var k=0; k<n; k++)
			tmp += vec[k] * mat[k][j];
		w[j] = tmp;
	}
	return w;	
}


function vectorize(f) {
	return function fv(v) {
		if (!Array.isArray(v)) 
			return f(v);
		else 
			return v.map(x => f(x));
	}
}
function sigmoid(x) { return 1. / (1.+Math.exp(-x)); }
var sigmoidv = vectorize(sigmoid);
var tanhv = vectorize(Math.tanh);

function dst(a, b) {
	var tmp = vmul(vsub(a, b), vsub(a, b));
	return tmp.reduce((acc, x) => acc+x);
}

function vadd(v1, v2) { return vop2(v1, v2, (a, b) => a+b) };
function vsub(v1, v2) { return vop2(v1, v2, (a, b) => a-b) };
function vmul(v1, v2) { return vop2(v1, v2, (a, b) => a*b) };

function vop2(v1, v2, op) {
	var n = v1.length,  
		n2 = v2.length;
	if (n!=n2) throw new Error("vop dimension mismatch");

	var w = new Array(n);
	for (var k=0; k<n; k++)
		w[k] = op(v1[k], v2[k]);
	return w;	
}

function vaddcste(v, c) {
	return v.map(x => x+c);	
}

test();