const fs = require('fs');

var txt = fs.readFileSync("input.js");
var n = txt.length;

console.log(n);

var m =0;
for (var i=0; i<n; i++)
	if (txt[i]>127) {
		txt[i] = 95; 	// '_'
		m++;
	}

console.log(m);

fs.writeFileSync("input.txt", txt);
