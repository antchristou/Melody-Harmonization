
init();

function getSelectedTrack(){
    var allTracks = trackMenu.getattr('items');
	if(typeof allTracks === 'string')
	{
		allTracks = [allTracks];
	}
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
	post("\nOUTPUT TRACK ITSELF",outputTrackMenu);
	var allTracks = outputTrackMenu.getattr('items');
	if(typeof allTracks === 'string')
	{
		allTracks = [allTracks];
	}
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
	if(typeof allClips === 'string')
	{
		allClips = [allClips];
	}
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

	var notes = get_notes(selectedClipID);
	
	post("\n clip dur "+notes.duration);
	post("\n clip first note "+notes.notes);
	
	// get key signature to transpose to C
	var keyValue = keySig.getvalueof();
	// format notes into form clips expect 
	model_input_notes = process_input(notes.notes,keyValue,notes.duration);
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
	post("AT TIME OF CLIP CREATION OUTPUT CLIP ID:",clip_id);
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
		notes["notes"][i]["pitch"] = notes["notes"][i]["pitch"]+transposeDistance;
	}
	clip.call("add_new_notes", notes);
	
	post("\n CREATED CLIP");
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
function process_input(notes,key_sig,clip_dur)
{
	
	post("Key sig: ", key_sig);
	var transposeDistance = distanceToC(parseInt(key_sig));
	post("\n Transpose value: ", transposeDistance);
	// edge case: if clip is totally empty, make rest for duration of clip
	post("\nWHAT WE GOT", notes,clip_dur)
	if (notes[1] == 0)
	{
		post("\n CLIP WITH NO NOTES DETECTED");
		return [["rest",clip_dur*4]];
	}
	
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
	
	// correct input that is polyphonic, or contains too long/short note values 
	// or contains rests 
	sorted_notes = correct_input(sorted_notes);
	
	input_notes = [];
	// start i at two to get past first two tokens which are word 'notes' and number of notes
	for(var i =0; i < sorted_notes.length; i = i+1)
	{
		// multiply by 4 to get duration in 16ths 
		if(sorted_notes[i][1] != "rest")
		{
		var noteDur = [sorted_notes[i][1]+transposeDistance,sorted_notes[i][3]*4];
		}
		else 
		{
		var noteDur = [sorted_notes[i][1],sorted_notes[i][3]*4];
		}
		input_notes.push(noteDur);
		
	}
	return input_notes;
	
}
function correct_input(sorted_notes)
{
	post("\n pre corrected notes " + sorted_notes);
	
	// notes array at this point is sorted and takes form of 
	// 'note', midi_pitch, start_time,
    // duration, velocity, mute status

	timesFilled = [];
		
	formattedNotes = [];

	// handle overlapping notes
	post("\nTHERE ARE THIS MANY NOTES", sorted_notes.length);
	for (var i = 0; i < sorted_notes.length; i++)
	{
		post("\nNEW",i, sorted_notes[i]);
		var overlap = has_overlap(timesFilled,sorted_notes[i])
		if(overlap == false)
		{
			formattedNotes.push(sorted_notes[i]);
			timesFilled.push(sorted_notes[i]);
		}
	}
	
	var noSmallNotes = [];
	for (var i = 0; i < formattedNotes.length; i++)
	{
		// remove notes that are non divisible by 16ths
		if(formattedNotes[i][3] % 0.25 == 0)
		{
			noSmallNotes.push(formattedNotes[i]);
		}
		else if(formattedNotes[i][3] > 0.25)
		{
			// round to the nearest multiple of 0.25
			post("\nROUNDING NOTE DOWN", formattedNotes[i][3]);
			var newDuration = Math.floor(formattedNotes[i][3] / 0.25) * 0.25;
			var newNote = [formattedNotes[i][0],formattedNotes[i][1],formattedNotes[i][2], newDuration, 
			formattedNotes[i][4],formattedNotes[i][5]];
			noSmallNotes.push(newNote);
		}
		else
		{
			post("\nSMALL NOTE DETECTED: ", formattedNotes[i][3]);
		}
	}
	formattedNotes = noSmallNotes;
	

	
	withRests = []
	// extrapolate rests in missing points
	for (var i = 0; i < formattedNotes.length-1; i++)
	{
		withRests.push(formattedNotes[i]);
		var previousEnd = formattedNotes[i][2]+formattedNotes[i][3];
		// if end time of previous note does not equal start time of previous, must add rest
		if(previousEnd != formattedNotes[i+1][2])
		{
			// round spaces in melody to nearest 16th note 
			// NOTE: this introduces some potential distorion of the length of the melody if this happens enough
			var restDur = Math.floor((formattedNotes[i+1][2]-previousEnd)/0.25)*0.25;
			var addedRest = ['note', 'rest',previousEnd, restDur];
			withRests.push(addedRest);
			post("\nADDING REST", addedRest);
		}
	}
	withRests.push(formattedNotes[formattedNotes.length-1]);
	// edge case if beginning would be a rest
	if(withRests[0][2] != 0)
	{
		var restDur = Math.floor(withRests[0][2]/0.25)*0.25;
		var addedRest = ['note', 'rest',0, restDur];
		withRests.splice(0,0,addedRest);
		post("\nADDING START REST", addedRest);
	}	
	

	
	formattedNotes = withRests;
	
	// cut off notes past 8 bars
	var totalLength = 0;
	for (var i = 0; i < formattedNotes.length; i++)
	{
		totalLength = totalLength + formattedNotes[i][3];
		post("\nTHE TOTAL LENGTH IS",totalLength);
		if (totalLength > 32)
		{
			post("\nREACHED MAX MELODY LENGTH");
			formattedNotes.splice(i);
			break;
		}
			
	}
		
	return formattedNotes;
}

function has_overlap(existing_notes, new_note)
{
	var does_overlap = false;

    for (var j = 0; j < existing_notes.length; j++){
		
	
		var new_start = new_note[2];
		var new_end = new_note[2]+new_note[3];
		
		var existing_start = existing_notes[j][2];
		var existing_end = existing_notes[j][2] + existing_notes[j][3];
		
	//	post("\nEXISTING",existing_notes[i]);
		
	//	post("\nEXISTING end",existing_start, existing_end);
	//	post("\n NEW end",new_start, new_end);
		
		// check note start vs note end (note start+note duration)
		if((new_start >= existing_start) && (new_end <= existing_end))
		{does_overlap = true}
		else if(new_start < existing_end && new_end > existing_end)
		{does_overlap = true}

	}
	post("\nIS THERE A OVERLAP WE ASK?",does_overlap);
    return does_overlap;
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
	post("\nGetting ALL CLIPS FROM SELECTED OUT", selectedTrack);
	var selectedTrackID = null;
	var tracks = getTracks();
	
	// get selected track ID from list of tracks 
	for (var i = 0; i < tracks.length; i++) 
	{ 
		post("output track NAME: "+selectedTrack + "\n");
		post(tracks[i].name+ "\n");
		if(tracks[i].name == selectedTrack){
			selectedTrackID = tracks[i].id;
			} 
	}
	
	post("\nOUTPUT Selected TRACK ID", selectedTrackID);
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
	post("\nYEEE");
	outlet(0,clipNames);
	
}

function getOutputClipID()
{
	//var allClips = getAllClipsFromSelectedOutputTrack();
	
	var allClips = outputClipMenu.getattr('items');
	// if only one clip cast as a list
	if(typeof allClips === 'string')
	{
		allClips = [allClips];
	}
    var b = outputClipMenu.getvalueof();

	post("\nWTF IS A B ",b);
	
	
    var selectedClip = allClips[b*2];
	
	// contains array of all clips and IDs for given output track
	var clipsArray  = getAllClipsFromSelectedOutputTrack()
	
	var selectedClipID;
	post("\n ALL CLIPS",allClips+"\n");
	post("\n TYPEOF ALL",typeof(allClips));

	// search to get id of selected clip 
	for (var i = 0; i < clipsArray.length; i++) 
	{ 
		post("\nsearching output...",clipsArray[i].name,selectedClip);
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
		post("\nThe selected track is: ",selectedTrack + "\n");
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
	post("\nHMMM");
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

function postAllIDs()
{
	var clips = getAllClips();
	for (i =0;i < clips.length;i++)
	{
		post("\nID:",clips[i].id);
	}
	
}

function postAllOutputIDs()
{
	var clips = getAllClipsFromSelectedOutputTrack();
	for (i =0;i < clips.length;i++)
	{
		post("\nID:",clips[i].id);
	}
	
	
}

function init(){
	
trackMenu = this.patcher.getnamed("trackMenu");
outputTrackMenu =  this.patcher.getnamed("outputTrackMenu");

clipMenu =  this.patcher.getnamed("clipMenu");
 outputClipMenu =  this.patcher.getnamed("outputClipMenu");

tempSlider = this.patcher.getnamed("tempSlider");

kSlider = this.patcher.getnamed("kSlider");

keySig = this.patcher.getnamed("keySig");
	}
