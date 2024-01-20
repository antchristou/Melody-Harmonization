

const maxApi = require('max-api');

const {spawn} = require('child_process');

maxApi.addHandler('run_model', (...input) => {
	maxApi.post("Received this "+input);
	var scriptPath = 'melody_harmonizer.py';
	const pythonExec = '/opt/homebrew/opt/python@3.11/bin/python3.11';
//	var scriptPath = 'testing.py';
//	const pyProg = spawn(pythonExec, [scriptPath,inputString]);
	//const pyProg = spawn(pythonExec, ['--version']);
	maxApi.post(typeof input);
	var modelIn = input[0];//"[[60,4],[60,4],[60,4]]";
	var temperature = input[1];
	var k_value = input[2]; 
	maxApi.post("temp: "+temperature); 
	maxApi.post("k value: "+k_value); 
	maxApi.post(modelIn);
	const pyProg = spawn(pythonExec, [scriptPath,"--daw",temperature, k_value, modelIn]);
	
//	maxApi.post("val for slider ", tempSlider.getvalueof());

	pyProg.stdout.on('data', function(data) {
	
	
	// convert output back to javascript dictionary 

	data = JSON.parse(data)
	data = JSON.stringify(data);
	maxApi.post("output from python: \n"+data);
	maxApi.outlet(data);
	});
//	maxApi.post("donez "+harmony);
	
	//maxApi.oulet(harmony);
	

//	maxApi.post("done "+data);
	// from here: get output chords in form useful for creating new clips
	// create new clips 
	
	pyProg.stderr.on('data', (data) => {
		maxApi.post("error on script call "+data);
	});
	
	pyProg.on('error', (err) => {
    console.error(`Error starting Python process: ${err}`);
});
	
});