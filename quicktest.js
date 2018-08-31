'use strict';

function model_from_data(data) {
	var model = {
		num_layers: data["num_layers"],
		rnn_size: data["rnn_size"],
		
		// convert text input unitx (chars, words, tokens into number)
		text_to_index : data["vocab"],
		index_to_text : data["chars"],
		
		// embedding of wordinto a n-dimension space (using t-SNE, PCA, ...)
		embedding: check_dim(data["embedding"]),
		
		// RNN layers
		rnn_layers : [],
		
		// output layer
		softmax_w: data["softmax_w"],
		softmax_b: data["softmax_b"],
		};
		
	var vars = data["cell_variables"];
	for (var i=0; i<vars.length; i+=2) {
		model.rnn_layers.push({
			w: check_dim(vars[i]), 
			b: vars[i+1].map((vL, iL) => check_dim(vL)) });
	}
	
	return model;
}

function check_dim(data) {
	if (data.length==1 && data[0].length==1)
		return data[0][0];
	if (data.length==1)
		return data[0];
	else
		return data;
}

function model_initial_state(num_layers, rnn_size) {
	var rnn_layer_states = [];
	for (var iL=0; iL<num_layers; iL++) 
		rnn_layer_states.push({ 
			"h": new Array(rnn_size).fill(0.0), 
			"c": new Array(rnn_size).fill(0.0) });
		
	var next_char_probs = null; // no input at that stage
			
	return { rnn_layer_states, next_char_probs };
}

function model_next_iter(model, iterations, input_char) {
	var nb_iters = iterations.length;
	
	// embedding
	var input_idx = model.text_to_index[input_char]
	var x = model.embedding[input_idx];
	//console.log("embedding x for input '"+input_char+"' ("+input_idx+") : ", x);
	
	// executed RNN layer stack
	var prev_iter = iterations[nb_iters-1];
	var next_iter = { 
		input: input_char, 
		rnn_layer_states: [] };
	for (var iL=0; iL<model.num_layers; iL++) {
		// previous state for this layer
		var { h, c } = prev_iter.rnn_layer_states[iL];
		// execute one iter on this layer
		var { h, c } = lstm_cell(x, h, c, model.rnn_layers[iL]);
		// save for next iteration
		next_iter.rnn_layer_states.push({ h, c });
		// input of next layer is output of layer below
		x = h;
	}
		
	// get probabilities
	next_iter.next_char_probs = softmax_output_layer_to_next_character(h, model);
	
	// save for next iteration
	iterations.push(next_iter);
	return next_iter;
}

// GITHUB Tensorflow implementation
// https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/rnn_cell_impl.py
function lstm_cell(x, h, c, lstm_prms) {
	var n = x.length;
	
	/* GITHUB 
	#  c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
	#  m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
	#  inputs, m_prev
	*/
	var v = [].concat(x, h);
	//console.log("vconcat")
	//console.log(shape(v))

	var w = vecmatmul(v, lstm_prms.w)
	var wb = vadd(w, lstm_prms.b)
	//console.log("vmultadd")
	//console.log(shape(wb))

	//  GITHUB    
	// i = input_gate, j = new_input, f = forget_gate, o = output_gate
	var wbi = wb.slice(0,n),
		wbc = wb.slice(n,2*n),
		wbf = wb.slice(2*n, 3*n),
		wbo = wb.slice(3*n);
	
	//console.log("vsplit", shape(wb), wbo, wbf, wbi, wbc)

	// GITHUB line.857  (as of commit 4134f74 on July 10th)
	// c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))
	var _forget_bias = 1.0;
	
	var o = sigmoidv(wbo),
		f = sigmoidv(vaddcste( wbf, _forget_bias)), 
		i = sigmoidv(wbi),
		ctmp = tanhv(wbc);
		
	//console.log("vsplit", o, f, i, ctmp)

	var cprev = c
	var cnext = vadd( vmul(cprev, f),  vmul(i, ctmp) )

	var hnext = vmul(tanhv(cnext), o)

	return { h: hnext, c: cnext };
}

function test_node() {
	var fs = require('fs');
	var data = JSON.parse(fs.readFileSync("save/js/model_data.json"));
	var model = model_from_data(data);
	var nL = model.num_layers;

	var inputE = check_dim(data["iterations"][0]["input_embedded"]),
		inputS = check_dim(data["iterations"][0]["input_squeezed"]),
		inputS2 = check_dim(data["iterations"][1]["input_squeezed"]),
		initC = check_dim(data["iterations"][0]["init_state_c"]),
		initH = check_dim(data["iterations"][0]["init_state_h"]),
		finalC = check_dim(data["iterations"][0]["final_state_c"]),
		finalH = check_dim(data["iterations"][0]["final_state_h"]),
		finalC2 = check_dim(data["iterations"][1]["final_state_c"]),
		finalH2 = check_dim(data["iterations"][1]["final_state_h"]);

	console.log("num_layers:", nL)
	console.log("text (Python):\n", data.text)
	
	console.log("input")
	nprint(inputE)
	nprint(inputS)
	console.log("state")
	nprint(initC)
	nprint(initH)
	nprint(finalC)
	nprint(finalH)
	
	console.log("\n ---- first iter ---- \n");

	var txt = "";
	var states = [];
	
	var input_char = data.prime[0]
	var x = data.embedding[data.vocab[input_char]];
	console.log("embedding x for input '"+input_char+"' ("+data.vocab[input_char]+") : ", x.slice(0,5));
	var x = inputS;
	console.log("embedding x expected : ", x.slice(0,5));
	
	var iL = 0;
	var [ h, c ] = nL==1 ? [ initH, initC ] : [ check_dim(initH[iL]), check_dim(initC[iL]) ];
	stepprint(x, h, c)
	var lstm_prms = model.rnn_layers[iL];
	var { h, c } = lstm_cell(x, h, c, lstm_prms);
	states.push({ h, c });
	console.log("errc", dst(c, nL==0 ? finalC : check_dim(finalC[0])));
	console.log("errh", dst(h, nL==0 ? finalH : check_dim(finalH[0])))
	
	if (nL==2) {
	
	var x = h; // input is output of previous layer
	
	var iL = 1;
	var [ h, c ] = nL==1 ? [ initH, initC ] : [ check_dim(initH[iL]), check_dim(initC[iL]) ];
	stepprint(x, h, c)
	var lstm_prms = model.rnn_layers[iL];
	var { h, c } = lstm_cell(x, h, c, lstm_prms);
	states.push({ h, c });
	console.log("errc", dst(c, nL==0 ? finalC : check_dim(finalC[nL-1])));
	console.log("errh", dst(h, nL==0 ? finalH : check_dim(finalH[nL-1])))
	
	}
	
	var probs = softmax_output_layer_to_probs(h, model);
	var next_char_probs = softmax_output_layer_to_next_character(h, model);
	var next_char = next_char_probs[0].c;
	txt += next_char;
	
	console.log("errp", dst(probs, check_dim(data.iterations[0].probs)))
	console.log();
	console.log("next_char_probs", next_char_probs.slice(0,10))
	console.log("next_char (max likelyhood pick)", next_char_probs[0].c);
	console.log("next_char (random pick)", pick_in_cumulative(next_char_probs));
	
	console.log("\n ---- subsequent iters ---- \n");

	for (var k=0; k<380;k++) {
		
		input_char = next_char;
		x = data.embedding[data.vocab[input_char]];
		
		var prev_states = states;
		var states = [];
		for (var iL=0; iL<nL; iL++) {
			
			var lstm_prms = model.rnn_layers[iL];
			var { h, c } = prev_states[iL];
			var { h, c } = lstm_cell(x, h, c, lstm_prms) ;
			states.push({ h, c });
		
			x = h;
		}
		
		var prev_char = next_char;
		var next_char_probs = softmax_output_layer_to_next_character(h, model);
		var next_char = ([ " ", "\n", "\t", "{", "}", "(", ")", "[", "]" ].indexOf(prev_char)!=-1)  
			? pick_in_cumulative(next_char_probs)
			: next_char_probs[0].c ;
		
		txt += next_char;
	}
	
	console.log("\n\n ---- GENERATED TEXT ---- \n" + txt + "\n")
	return txt;
	
	function nprint(x) {
		console.log(shape(x));
	}

	function stepprint(x, h, c) {
		console.log()
		console.log ("x", shape(x), x.slice(0,5))
		console.log ("h", shape(h), h.slice(0,5))
		console.log ("c", shape(c), c.slice(0,5))
	}
}


function softmax_output_layer_to_probs(h, model) {
	var w = vecmatmul(h, model.softmax_w);
	var wb = vadd(w, model.softmax_b);
	
	var exp_wb = expv(wb);
	var sum_exp_wb = exp_wb.reduce((acc,x) => acc+x);
	var probs = vmulcste(exp_wb, 1.0/sum_exp_wb);
	return probs;
}

function softmax_output_layer_to_next_character(h, model) {
	var probs = softmax_output_layer_to_probs(h, model);
	var idx_to_char = model.index_to_text;
	var char_prob = probs.map((prob, idx) => { return { c: idx_to_char[idx], prob: prob } });
	var char_prob_top = char_prob.sort((a,b) => b.prob-a.prob) // .slice(0,10);
	//console.log(char_prob_top);
	return char_prob_top;
}

function pick_in_cumulative(next_char_probas) {
	var unif = Math.random();
	var cumul = 0;
	for (var nc of next_char_probas) {
		cumul += nc.prob;
		if (unif<cumul) return nc.c;
	}
	console.log("UNLIKELY pick_in_cumulative, unif="+unif+", cumul up to "+cumul);
	return "!";
}

// defining few NumPy utility fucntions 

function shape(a) {
	if (a.length==0 || !Array.isArray(a[0])) {
		return [a.length];
	} else {
		var tmp = shape(a[0]);
		tmp.unshift(a.length);
		return tmp;
	}
}

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
var expv = vectorize(Math.exp);

function dst(a, b) {
	var tmp = vmul(vsub(a, b), vsub(a, b));
	return tmp.reduce((acc, x) => acc+x);
}

function vadd(v1, v2) { return vop2(v1, v2, (a, b) => a+b) };
function vsub(v1, v2) { return vop2(v1, v2, (a, b) => a-b) };
function vmul(v1, v2) { return vop2(v1, v2, (a, b) => a*b) };
function vdiv(v1, v2) { return vop2(v1, v2, (a, b) => a/b) };

function vop2(v1, v2, op) {
	var n = v1.length,  
		n2 = v2.length;
	if (n!=n2) throw new Error("vop dimension mismatch." + n + " vs" + n2);

	var w = new Array(n);
	for (var k=0; k<n; k++)
		w[k] = op(v1[k], v2[k]);
	return w;	
}

function vaddcste(v, c) { return v.map(x => x+c); }
function vmulcste(v, c) { return v.map(x => x*c); }

if (typeof module !== 'undefined' && module.exports) {
	test_node();
}
