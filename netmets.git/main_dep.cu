#include <stdlib.h>
#include <string>
#include <fstream>
#include <algorithm>

//OpenGL includes
#include <GL/glut.h>
#include <GL/freeglut.h>

//STIM includes
#include <stim/visualization/gl_network.h>
#include <stim/biomodels/network.h>
#include <stim/visualization/gl_aaboundingbox.h>
#include <stim/parser/arguments.h>
#include <stim/visualization/camera.h>

#ifdef __CUDACC__
//CUDA includes
#include <cuda.h>
#endif

//ANN includes
//#include <ANN/ANN.h>

//BOOST includes
#include <boost/tuple/tuple.hpp>

//visualization objects
stim::gl_aaboundingbox<float> bb;			//axis-aligned bounding box object
stim::camera cam;					//camera object

unsigned num_nets = 0;
stim::gl_network<float> GT;			//ground truth network
stim::gl_network<float> T;			//test network

//hard-coded parameters
float resample_rate = 0.5f;			//sample rate for the network (fraction of sigma used as the maximum sample rate)
float camera_factor = 1.2f;			//start point of the camera as a function of X and Y size
float orbit_factor = 0.01f;			//degrees per pixel used to orbit the camera

//mouse position tracking
int mouse_x;
int mouse_y;

//OpenGL objects
GLuint cmap_tex = 0;				//texture name for the color map

//sets an OpenGL viewport taking up the entire window
void glut_render_single_projection(){

	glMatrixMode(GL_PROJECTION);					//load the projection matrix for editing
	glLoadIdentity();								//start with the identity matrix
	int X = glutGet(GLUT_WINDOW_WIDTH);				//use the whole screen for rendering
	int Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, X, Y);							//specify a viewport for the entire window
	float aspect = (float)X / (float)Y;				//calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		//set up a perspective projection
}

//sets an OpenGL viewport taking up the left half of the window
void glut_render_left_projection(){

	glMatrixMode(GL_PROJECTION);					//load the projection matrix for editing
	glLoadIdentity();								//start with the identity matrix
	int X = glutGet(GLUT_WINDOW_WIDTH) / 2;			//only use half of the screen for the viewport
	int Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(0, 0, X, Y);							//specify the viewport on the left
	float aspect = (float)X / (float)Y;				//calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		//set up a perspective projection
}

//sets an OpenGL viewport taking up the right half of the window
void glut_render_right_projection(){

	glMatrixMode(GL_PROJECTION);					//load the projection matrix for editing
	glLoadIdentity();								//start with the identity matrix
	int X = glutGet(GLUT_WINDOW_WIDTH) / 2;			//only use half of the screen for the viewport
	int Y = glutGet(GLUT_WINDOW_HEIGHT);
	glViewport(X, 0, X, Y);							//specify the viewport on the right
	float aspect = (float)X / (float)Y;				//calculate the aspect ratio
	gluPerspective(60, aspect, 0.1, 1000000);		//set up a perspective projection
}

void glut_render_modelview(){

	glMatrixMode(GL_MODELVIEW);						//load the modelview matrix for editing
	glLoadIdentity();								//start with the identity matrix
	stim::vec3<float> eye = cam.getPosition();		//get the camera position (eye point)
	stim::vec3<float> focus = cam.getLookAt();		//get the camera focal point
	stim::vec3<float> up = cam.getUp();				//get the camera "up" orientation

	gluLookAt(eye[0], eye[1], eye[2], focus[0], focus[1], focus[2], up[0], up[1], up[2]);	//set up the OpenGL camera
}

//draws the network(s)
void glut_render(void) {

	if(num_nets == 1){											//if a single network is loaded
		glut_render_single_projection();						//fill the entire viewport
		glut_render_modelview();								//set up the modelview matrix with camera details
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		//clear the screen
		GT.glCenterline(GT.nmags() - 1);						//render the GT network (the only one loaded)
	}

	if(num_nets == 2){											//if two networks are loaded	

		glut_render_left_projection();							//set up a projection for the left half of the window
		glut_render_modelview();								//set up the modelview matrix using camera details
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		//clear the screen

		glEnable(GL_TEXTURE_1D);										//enable texture mapping
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);		//texture map will be used as the network color
		glBindTexture(GL_TEXTURE_1D, cmap_tex);							//bind the Brewer texture map

		GT.glCenterline(GT.nmags() - 1);						//render the GT network

		glut_render_right_projection();							//set up a projection for the right half of the window
		glut_render_modelview();								//set up the modelview matrix using camera details
		T.glCenterline(T.nmags() - 1);							//render the T network

	}

	glutSwapBuffers();
}

// defines camera motion based on mouse dragging
void glut_motion(int x, int y){
	

	float theta = orbit_factor * (mouse_x - x);		//determine the number of degrees along the x-axis to rotate
	float phi = orbit_factor * (y - mouse_y);		//number of degrees along the y-axis to rotate

	cam.OrbitFocus(theta, phi);						//rotate the camera around the focal point

	mouse_x = x;									//update the mouse position
	mouse_y = y;
		
	glutPostRedisplay();							//re-draw the visualization
}

// sets the mouse position when clicked
void glut_mouse(int button, int state, int x, int y){
	mouse_x = x;
	mouse_y = y;
}

#define BREWER_CTRL_PTS 11							//number of control points in the Brewer map
void texture_initialize(){

	//define the colormap
	static float  brewer_map[BREWER_CTRL_PTS][3] = {			//generate a Brewer color map (blue to red)
		{0.192157f, 0.211765f, 0.584314f},
		{0.270588f, 0.458824f, 0.705882f},
		{0.454902f, 0.678431f, 0.819608f},
		{0.670588f, 0.85098f, 0.913725f},
		{0.878431f, 0.952941f, 0.972549f},
		{1.0f, 1.0f, 0.74902f},
		{0.996078f, 0.878431f, 0.564706f},
		{0.992157f, 0.682353f, 0.380392f},
		{0.956863f, 0.427451f, 0.262745f},
		{0.843137f, 0.188235f, 0.152941f},
		{0.647059f, 0.0f, 0.14902f}
	};

	glGenTextures(1, &cmap_tex);								//generate a texture map name
	glBindTexture(GL_TEXTURE_1D, cmap_tex);						//bind the texture map

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);		//enable linear interpolation
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);			//clamp the values at the minimum and maximum
	glTexImage1D(GL_TEXTURE_1D, 0, 3, BREWER_CTRL_PTS, 0, GL_RGB, GL_FLOAT,	//upload the texture map to the GPU
					brewer_map);
}

//Initialize the OpenGL (GLUT) window, including starting resolution, callbacks, texture maps, and camera
void glut_initialize(){
	
	int myargc = 1;					//GLUT requires arguments, so create some bogus ones
	char* myargv[1];
	myargv [0]=strdup ("netmets");

	glutInit(&myargc, myargv);									//pass bogus arguments to glutInit()
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);	//generate a color buffer, depth buffer, and enable double buffering
	glutInitWindowPosition(100,100);							//set the initial window position
	glutInitWindowSize(320,320);								//set the initial window size
	glutCreateWindow("NetMets - STIM Lab, UH");					//set the dialog box title

	
	// register callback functions
	glutDisplayFunc(glut_render);			//function executed for rendering - renders networks
	glutMouseFunc(glut_mouse);				//executed on a mouse click - sets starting mouse positions for rotations
	glutMotionFunc(glut_motion);			//executed when the mouse is moved while a button is pressed

	texture_initialize();					//set up texture mapping (create texture maps, enable features)

	stim::vec3<float> c = bb.center();		//get the center of the network bounding box

	//place the camera along the z-axis at a distance determined by the network size along x and y
	cam.setPosition(c + stim::vec<float>(0, 0, camera_factor * std::max(bb.size()[0], bb.size()[1])));
	cam.LookAt(c[0], c[1], c[2]);						//look at the center of the network

	glClearColor(1, 1, 1, 1);
}

#ifdef __CUDACC__
void setdevice(int &device){
	int count;
	cudaGetDeviceCount(&count);				// numbers of device that are available
	if(count < device + 1){
	std::cout<<"No such device available, please set another device"<<std::endl;
	exit(1);
	}
}
#else
void setdevice(int &device){
	device = -1;
}
#endif

//compare both networks and fill the networks with error information
void compare(float sigma, int device){

	GT = GT.compare(T, sigma, device);						//compare the ground truth to the test case - store errors in GT
    T = T.compare(GT, sigma, device);						//compare the test case to the ground truth - store errors in T

	//calculate the metrics
	float FPR = GT.average(0);						//calculate the metrics
	float FNR = T.average(0);
	
	std::cout << "FNR: " << FPR << std::endl;		//print false alarms and misses
	std::cout << "FPR: " << FNR << std::endl;
}

// writes features of the networks i.e average segment length, tortuosity, branching index, contraction, fractal dimension, number of end and branch points to a csv file
// Pranathi wrote this - saves network features to a CSV file
void features(std::string filename){
		double avgL_t, avgL_gt, avgT_t, avgT_gt, avgB_t, avgB_gt, avgC_t, avgC_gt, avgFD_t, avgFD_gt;
		unsigned int e_t, e_gt, b_gt, b_t;
		avgL_gt = GT.Lengths();
		avgT_gt = GT.Tortuosities();
		avgL_t = T.Lengths();
		avgT_t = T.Tortuosities();
		avgB_gt = GT.BranchingIndex();
		avgB_t = T.BranchingIndex();
		avgC_gt = GT.Contractions();
		avgFD_gt = GT.FractalDimensions();
		avgC_t = T.Contractions();
		avgFD_t = T.FractalDimensions();
		e_gt = GT.EndP();
		e_t = T.EndP();
		b_gt = GT.BranchP();
		b_t = T.BranchP();
		std::ofstream myfile;
		myfile.open (filename.c_str());
		myfile << "Length, Tortuosity, Contraction, Fractal Dimension, Branch Points, End points, Branching Index, \n";
		myfile << avgL_gt << "," << avgT_gt << "," << avgC_gt << "," << avgFD_gt << "," << b_gt << "," << e_gt << "," << avgB_gt <<std::endl;
		myfile << avgL_t << "," << avgT_t << "," << avgC_t << "," << avgFD_t << "," << b_t << "," << e_t << "," << avgB_t <<std::endl;
		myfile.close();
}

// Output an advertisement for the lab, authors, and usage information
void advertise(){
	std::cout<<std::endl<<std::endl;
	std::cout<<"========================================================================="<<std::endl;
	std::cout<<"Thank you for using the NetMets network comparison tool!"<<std::endl;
	std::cout<<"Scalable Tissue Imaging and Modeling (STIM) Lab, University of Houston"<<std::endl;
	std::cout<<"Developers: Pranathi Vemuri, David Mayerich"<<std::endl;
	std::cout<<"Source: https://git.stim.ee.uh.edu/segmentation/netmets"<<std::endl;
	std::cout<<"========================================================================="<<std::endl<<std::endl;

	std::cout<<"usage: netmets file1 file2 --sigma 3"<<std::endl;
	std::cout<<"            compare two files with a tolerance of 3 (units defined by the network)"<<std::endl<<std::endl;
	std::cout<<"       netmets file1 --gui"<<std::endl;
	std::cout<<"            load a file and display it using OpenGL"<<std::endl<<std::endl;
	std::cout<<"       netmets file1 file2 --device 0"<<std::endl;
	std::cout<<"            compare two files using device 0 (if there isn't a gpu, use cpu)"<<std::endl<<std::endl;
}

int main(int argc, char* argv[])
{
	stim::arglist args;						//create an instance of arglist

	//add arguments
	args.add("help", "prints this help");
	args.add("sigma", "force a sigma value to specify the tolerance of the network comparison", "3");
	args.add("gui", "display the network or network comparison using OpenGL");
	args.add("device", "choose specific device to run", "0");
	args.add("features", "save features to a CSV file, specify file name");

	args.parse(argc, argv);					//parse the user arguments

	if(args["help"].is_set() || args.nargs() == 0){			//test for help
		advertise();										//output the advertisement
		std::cout<<args.str();								//output arguments
		exit(1);											//exit
	}
	
	if(args.nargs() >= 1){					//if at least one network file is specified
		num_nets = 1;						//set the number of networks to one
		GT.load_obj(args.arg(0));			//load the specified file as the ground truth
		/*GT.to_txt("Graph.txt");*/
	}
	
	if(args.nargs() == 2){			//if two files are specified, they will be displayed in neighboring viewports and compared
		int device = args["device"].as_int();				//get the device value from the user
		num_nets = 2;										//set the number of networks to two
		float sigma = args["sigma"].as_float();				//get the sigma value from the user
		T.load_obj(args.arg(1));                           //load the second (test) network
		if(args["features"].is_set())						//if the user wants to save features
			features(args["features"].as_string());
		GT = GT.resample(resample_rate * sigma);			//resample both networks based on the sigma value
		T = T.resample(resample_rate * sigma);
		setdevice(device);
		compare(sigma, device);										//run the comparison algorithm
	}

	//if a GUI is requested, display the network using OpenGL
	if(args["gui"].is_set()){		
		bb = GT.boundingbox();					//generate a bounding volume		
		glut_initialize();						//create the GLUT window and set callback functions		
		glutMainLoop();							// enter GLUT event processing cycle
	}	
}
