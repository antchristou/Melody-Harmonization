
var trackMenu = this.patcher.getnamed("trackMenu");
var outputTrackMenu =  this.patcher.getnamed("outputTrackMenu");

var clipMenu =  this.patcher.getnamed("clipMenu");
var outputClipMenu =  this.patcher.getnamed("outputClipMenu");

var tempSlider = this.patcher.getnamed("tempSlider");

var kSlider = this.patcher.getnamed("kSlider");

var keySig = this.patcher.getnamed("keySig");

function getSelectedTrack(){
    var allTracks = trackMenu.getattr('items');
    var b = trackMenu.getvalueof();

	// todo: investigate why x2 works heh..
    var selectedTrack = allTracks[b*2];
	post("selecting" + selectedTrack + "\n");
	return selectedTrack;
	// TODO: disable if not selected 
	//if(typeof(selectedTrack) == null)
    }

function getSelectedOutputTrack()
{ 
	var allTracks = outputTrackMenu.getattr('items');
    var b = outputTrackMenu.getvalueof();

	// todo: investigate why x2 works heh..
    var selectedTrack = allTracks[b*2];
	post("selecting" + selectedTrack + "\n");
	return selectedTrack;
	// TODO: disable if not selected 
	//if(typeof(selectedTrack) == null)
}

function getSelectedClip()
{ 
	var allClips = clipMenu.getattr('items');
    var b = clipMenu.getvalueof();

	// todo: investigate why x2 works heh..
    var selectedClip = allClips[b*2];
	post("selecting" + selectedClip + "\n");
	return selectedClip;
	// TODO: disable if not selected 
	//if(typeof(selectedTrack) == null)
}


// using notes from selected input clip, output harmony to selected output destination
function Harmonize()
{
	// get id from selected clip 
	var selectedClip = getSelectedClip();
	
	var selectedClipID = null;
		
	// TODO: make sure things are actually selected 
	
	var allClips = getAllClips();
	post(allClips.length);
	
	// get selected clip ID from list of clips 
	for (var i = 0; i < allClips.length; i++) 
	{ 
		if(allClips[i].name == selectedClip && !allClips[i].empty){
			selectedClipID = allClips[i].id;
			} 
	}
	// get notes from selected clip 
	// clip has following attributes: id of clip, notes (0-127 in midi), and duration of clip
	// TODO: make sure clip is less than max 8 bars 

	var notes = get_notes(selectedClipID);
	
	post("\n clip dur "+notes.duration);
	post("\n clip first note "+notes.notes);
	
	// get key signature to transpose to C
	var keyValue = keySig.getvalueof();
	// format notes into form clips expect 
	model_input_notes = process_input(notes.notes,keyValue);
	post("\nPOSTING "+model_input_notes);
	
	// pass into python machine learning model 
	var output = run_model(model_input_notes);
	
	// turn back into ableton notes format, output to clip
	
	
}


function create_clip(notes)
{
	post("printing input 1 " + notes);
	notes = JSON.parse(notes);
	post("printing input 2 \n" + notes["notes"][0]["pitch"]);
	
		
//	post(Object.keys(notes));
	var clip = new LiveAPI()
	// get currently selected output clip 
	clip_id = getOutputClipID();
	clip.id = clip_id
	if (clip.type === 'ClipSlot'){
		clip.id = clip.get('clip')[1]
	}
	
	post("output clip id", clip_id);
	
	// see docs here on how notes dictionary should look: 
	// https://docs.cycling74.com/max8/vignettes/live_object_model
	//var notes = {notes: [{
	//	pitch : 60,
	//	start_time : 0,
	//	duration : 4,
//	}]};
	clip.call("remove_notes_extended", 0,127,0,9999);
	
	// transpose back to original key - inversion of original distance
	var transposeDistance = distanceToC(parseInt(keySig.getvalueof()))*-1;
	post("\n Transposing back by: ", transposeDistance);
	
	for(i = 0; i < notes["notes"].length;i++)
	{
		post("\n Pre final transpose note value: ", notes["notes"][i]["pitch"]);
		notes["notes"][i]["pitch"] = notes["notes"][i]["pitch"]+transposeDistance;
	}
	clip.call("add_new_notes", notes);
}

// executes python script
function run_model(model_input_notes)
{
	
	var inputString = JSON.stringify(model_input_notes);
	
	var temp = parseFloat(tempSlider.getvalueof());
	
	var k_val = parseInt(kSlider.getvalueof());
	
	var inputList = [inputString, temp, k_val]; 
	outlet(0,inputList);

}
// prepare raw API call for input to model
// model expects form of tuples [note,duration in 16ths]
function process_input(notes,key_sig)
{
	
	post("Key sig: ", key_sig);
	var transposeDistance = distanceToC(parseInt(key_sig));
	post("\n Transpose value: ", transposeDistance);
	
	// need to sort based on start time as api call isn't sorted 
	sorted_notes = []
	for(var i = 2; i < notes.length-1;i=i+6)
	{
		 sorted_notes.push(notes.slice(i, i + 6));
	}
	
	sorted_notes.sort(function(a, b) {
  	return a[2] - b[2];
	});
	
	post(sorted_notes[0]+" THEN!! " +sorted_notes[1]);
	
	input_notes = [];
	// start i at two to get past first two tokens which are word 'notes' and number of notes
	for(var i =0; i < sorted_notes.length; i = i+1)
	{
		// multiply by 4 to get duration in 16ths 
		var noteDur = [sorted_notes[i][1]+transposeDistance,sorted_notes[i][3]*4];
		input_notes.push(noteDur);
		
	}
	return input_notes;
	
}

function distanceToC(number)
{
	
  if (number == 0) {return 0;}
  const remainder = (number - 0)*-1 ;
  const complementaryRemainder = 12 - number;
  if (Math.abs(remainder) < complementaryRemainder) {
	return remainder;
}
  return complementaryRemainder
}

function getAllClipsFromSelectedOutputTrack()
{
	var selectedTrack = getSelectedOutputTrack();
	var selectedTrackID = null;
	var tracks = getTracks();
	
	// get selected track ID from list of tracks 
	for (var i = 0; i < tracks.length; i++) 
	{ 
		post("output track: "+selectedTrack + "\n");
		post(tracks[i].name+ "\n");
		if(tracks[i].name == selectedTrack){
			selectedTrackID = tracks[i].id;
			} 
	}
	
	post(selectedTrackID);
	post(typeof(selectedTrackID));
	var allClips = clips(selectedTrackID);
	return allClips
}

// get all clips
function populateOutputClipMenu()
{
	var allClips = getAllClipsFromSelectedOutputTrack();
	
	var clipNames = [];
	
	for (var j = 0; j < allClips.length; j++)
	{
		if(!allClips[j].empty){
		clipNames.push(allClips[j].name);
		}
			
	}
	outlet(0,clipNames);
	
}

function getOutputClipID()
{
	//var allClips = getAllClipsFromSelectedOutputTrack();
	
	var allClips = outputClipMenu.getattr('items');
	
    var b = outputClipMenu.getvalueof();


    var selectedClip = allClips[b*2];
	
	// contains array of all clips and IDs for given output track
	var clipsArray  = getAllClipsFromSelectedOutputTrack()
	
	var selectedClipID;

	// search to get id of selected clip 
	for (var i = 0; i < clipsArray.length; i++) 
	{ 
		
		if(clipsArray[i].name == selectedClip && !clipsArray[i].empty){
			selectedClipID = clipsArray[i].id;
			} 
	}
	post("selected output clip idea "+ selectedClipID);
	return selectedClipID;
	
	
}

function getAllClips()
{
	
	var selectedTrack = getSelectedTrack();
	var selectedTrackID = null;
	var tracks = getTracks();
	
	// get selected track ID from list of tracks 
	for (var i = 0; i < tracks.length; i++) 
	{ 
		post(selectedTrack + "\n");
		post(tracks[i].name+ "\n");
		if(tracks[i].name == selectedTrack){
			selectedTrackID = tracks[i].id;
			} 
	}
	
	post(selectedTrackID);
	post(typeof(selectedTrackID));
	var allClips = clips(selectedTrackID);
	
	return allClips;
}

// get all clips that have midi data from selected track
function populateClipMenu()
{
	var allClips = getAllClips();
	post(allClips.length);
	
	var clipNames = [];
	
	for (var j = 0; j < allClips.length; j++)
	{
		if(!allClips[j].empty){
			clipNames.push(allClips[j].name);
			}
	}
	
	outlet(0,clipNames);

}
// get all currently existing tracks and clips for menus - needs to be done repeatedely 
function populateTrackMenu()
{
	var tracks = getTracks();
	var trackNames = tracks.map(function(tracks) {return tracks.name;});
	// send track names back to Max
	outlet(0,trackNames);
	
}



//get all tracks in project
function getTracks()
{
	live_api = new LiveAPI();
	live_api.path = 'live_set';
	var tracks = live_api.get('tracks');
	post(tracks.length);
	
	var retTracks = []
	for (var i = 0; i < tracks.length; i++){
		if (tracks[i] !== 'id'){
			var track = new LiveAPI()
			track.id = parseInt(tracks[i])
			if (parseInt(track.get('has_midi_input'))){
				retTracks.push({
					name : track.get('name')[0],
					id : parseInt(track.id),
				})
			}
		}
	}
//	outlet(0,retTracks);
	//outlet(0,"hi");
	//post(retTracks.length)
	//post(retTracks[0].name);
	//post(retTracks[0].id);
	
	return retTracks;
}

// getting all clips from track of some ID 
function clips(clip_id){
	var track = new LiveAPI()
	track.id = clip_id
	var clips = track.get('clip_slots')
	var retClips = []
	for (var j = 0; j < clips.length; j++){
		if (clips[j] !== 'id'){
			var clipSlot = new LiveAPI()
			clipSlot.id = clips[j]
			var hasClip = Boolean(clipSlot.get('has_clip')[0])
			var clipName = ''
			if (hasClip){
				var innerClip = new LiveAPI()
				innerClip.id = clipSlot.get('clip')[1]
				clipName = innerClip.get('name')[0]
			}
			retClips.push({
				id : clips[j],
				empty : !hasClip,
				name : clipName
			})
		}
	}
	return retClips;
	
	post(retClips[0].id);
}

// get notes from given clip

function get_notes(clip_id){
	var clip = new LiveAPI()
	clip.id = clip_id
	if (clip.type === 'ClipSlot'){
		clip.id = clip.get('clip')[1]
	}
	// 'notes' contains  pitch, time, duration, velocity, and mute state.
	// speficially formated as: 
    // word 'notes', num_notes in clip, word 'note', midi_pitch, start_time, duration, velocity, mute status
	var notes = {
		id : clip.id,
		notes : clip.call('get_notes', 0, 0, 10000, 127),
		duration : clip.get('end_time')[0]
	}
	return notes;
}
