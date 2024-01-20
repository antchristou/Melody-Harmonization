const maxApi = require('max-api');

const {spawn} = require('child_process');

maxApi.addHandler('run_model', (input) => {
	maxApi.post("Received this "+input);
	var scriptPath = 'melody_harmonizer.py';
	
	const pyProg = spawn('python3', [scriptPath,inputString]);
//	const pyProg = spawn('python3', ['--version']);

	pyProg.stdout.on('data', function(data) {
	maxApi.post("donez");
	});
	
	pyProg.stderr.on('data', (data) => {
		maxApi.post("error on script call");
	});
	
});