// read in files to "filesDir"
//dir = getDirectory("Choose a Directory");
//dir = "J:\\DATA_2017-2018\\Optic_nerve\\EAE_miR_AAV2\\2018.08.07\\ON_11\\ROIs\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Etienne\\Control Images\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Training Data\\New folder\\"
dir = getDirectory("Choose a Directory");
//setBatchMode(true);
// ***ALSO MUST OPEN AN IMAGE OF THE CORRECT SIZE WHICH NAME MATCHES LINE #96
//count = 0;

//list = getFileList(dir);
//Array.sort(list);
//for (i=0; i<list.length; i++) {
//     count++;
//	 print(list[i]);
//}
//n = 0;
//processFiles(dir);
//print(count + "files processed");

//add_color = 1;
//last_num_roi = 0;

getDimensions(width, height, channels, slices, frames);
name = getInfo("image.filename");
print(name);
print(frames);
for (i = 1; i <= frames; i+=1) {
		//run("Concatenate...", "open image1=[" + list[i] + "] image2=[" + list[i + 1] + "]");
		run("Make Substack...", "slices=1-60 frames=" + i);


		// Run correct 3 drift
		//run("Correct 3D drift", "1");

	
		// SAVE THE FILE
		//run("Make Substack...", "frames=1");
		num = "" + i;
		if (i < 10) {
			num = "00" + num;
		}
		else if (i < 100) {
			num = "0" + num;
		}

			
		tmpStr = substring(name, 0, lengthOf(name) - 4);
		sav_Name = tmpStr + "_slice_" + num + ".tif";
		saveAs("Tiff", dir + sav_Name);
					
		run("Close");
		call("java.lang.System.gc");    // clears memory leak
	 	call("java.lang.System.gc"); 
	  	call("java.lang.System.gc"); 
	  	call("java.lang.System.gc"); 
	    call("java.lang.System.gc"); 
	    call("java.lang.System.gc"); 
	    call("java.lang.System.gc"); 
	    call("java.lang.System.gc"); 
	    call("java.lang.System.gc"); 
	    call("java.lang.System.gc"); 

}

