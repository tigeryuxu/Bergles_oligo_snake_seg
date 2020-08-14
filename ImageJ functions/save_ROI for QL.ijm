/*
 * Reads name of current image and saves ROI in same working directory as line 13
 * 
 * First reads cwd and active image name
 * Splits by "." ***note that my names have TWO .tifs b/c I named them weirdly. change line 27 to "2" instead of "3" if only ONE ".tif" in name
 *
 * Then appends a numeric counter to the image_name to keep track of the cell number
 * thus, when saved, the ROI name will correlate with cell number
 * 
 * Finally, deletes your ROIs to return to clean slate
 */

cwd = "D:\\1) Myelin quantification old backup\\New Folder\\"
print(cwd);

// get name of current image
im_name = getTitle();
print(im_name);
splitted = split(im_name, ".");
first_name = splitted[0];

// new name of image to keep cell counter
print(splitted.length);
print(splitted[splitted.length - 1]);

new_name = "";
if (splitted.length == 2) {    // CHANGE TO 2 if only ONE ".tif" in image name
	x = 1;
	// create filename
	new_name = cwd + first_name + '_' + x + '.zip';
	
	new_rename = im_name + '.' + x;
	rename(new_rename);
	
}
else {
	num = splitted[splitted.length - 1];  // gets num idx of cell
	
	x = parseInt(num); // get number
	print(x);
	x = x + 1; // augment counter

	new_name = cwd + first_name + '_' + x + '.zip';

	chars_overhang = 1;  // accounts for how many final chars to take off, so doesn't explode exponentially
	if (x > 10) {
		chars_overhang = 2;
	}
	
	if (x > 100) {
		chars_overhang = 3;
	}

    im_name = substring(im_name, 0, lengthOf(im_name) - chars_overhang); // takes off final character(s)
	new_rename = im_name + x;
	rename(new_rename);
}

// save
roiManager("Save", new_name);
print(new_name);

// then delete all the ROIs so have clean slate
roiManager("delete");
