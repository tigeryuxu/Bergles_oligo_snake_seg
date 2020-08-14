// delete everything in ROImanager
for (index = 0; index < roiManager("count"); index++) {
	roiManager("delete");
	print(index);
}

// read in files to "filesDir"
dir = getDirectory("Choose a Directory");
//setBatchMode(true);
count = 0;

list = getFileList(dir);
for (i=0; i<list.length; i++) {
     count++;
	 print(list[i]);
}
n = 0;
//processFiles(dir);
print(count + "files processed");

add_color = 1;
last_num_roi = 0;
for (i = 0; i < list.length; i++) {   // augment by 5

	path = dir + list[i];
	print(path);
	//open(path);
	match = 0;
	if (i == 0) {
		match = 0;
	}
	else {
		cur_f = split(list[i - 1], "_");
		next_f = split(list[i], "_");
		
		if (cur_f[0] + cur_f[1] == next_f[0] + next_f[1]) {
			match = 1;
			print("Im in");
		}
	}

	// if they do match, then just open and append the ROI without saving the image
	if (match == 1) {
		roiManager("Open", path);
		add_color = add_color + 2;
	}
	else if (match == 0 && i == 0) {
		newImage("Labeling", "16-bit black", getWidth(), getHeight(), 1);		
		roiManager("Open", path);
	}
	
	else if (match == 0) {
		add_color = 1;
		// save image
		selectWindow("Labeling");
		print(dir + "Mask" + path);
		saveAs("Tiff", dir + "Mask_" + list[i - 1]);	
		close();		

	    // then delete everything in the ROI manager
		for (index = 0; index < roiManager("count"); index++) {
			roiManager("delete");
			print(index);
		}

		roiManager("Open", path);
		newImage("Labeling", "16-bit black", getWidth(), getHeight(), 1);
		
	}
	selectWindow("Labeling");
	// print with different indices
	if (add_color == 0) {   // if it's the first one, then index = 0
		index = 0;
	}
	else {
		index = last_num_roi;
		print("here");
	}
	last_num_roi = roiManager("count");
	first = 0;
	while (index < roiManager("count")) {
		roiManager("select", index);
		if (first == 0) {
			setColor(add_color);
		}
		else {
			setColor(add_color + 1);
		}
		fill();
		first++;
		index++;
	}
	resetMinAndMax();
	run("glasbey");

	//selectWindow("uFNet-04_2b-7d-Orthogonal Projection-01-Scene-01-uFNet-04_1-Max-09_c1+2.tif");
	selectWindow("180215W_uFNet-01_2b(23)-036_z5c1+3+4.tif_Z_PROJECT_only_4.tif");
	
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

// save image
selectWindow("Labeling");
print(dir + "Mask" + path);
saveAs("Tiff", dir + "Mask_" + list[i - 1]);	
close();	