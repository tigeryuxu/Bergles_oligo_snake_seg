#Specify the version being used as well as the language
cmake_minimum_required(VERSION 2.8.11)

#Name your project here
project(netmets)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#find the STIM library
find_package(STIM)

#if the STIM library isn't found, download it
if(NOT STIM_INCLUDE_DIRS)
	file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/stimlib)	#remove the stimlib directory if it exists
	set(STIM_GIT "https://git.stim.ee.uh.edu/codebase/stimlib.git")
	execute_process(COMMAND git clone --depth 1 ${STIM_GIT} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	set(STIM_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/stimlib" CACHE TYPE PATH)
endif(NOT STIM_INCLUDE_DIRS)

#find BOOST
find_package(Boost REQUIRED)

#find cuda
find_package(CUDA REQUIRED)

#find the GLUT library for visualization
find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

#GLEW is required for MS Visual C
if(MSVC)
	find_package(GLEW REQUIRED)
	include_directories(${GLEW_INCLUDE_DIR})
endif(MSVC)

#find OpenCV
find_package(OpenCV REQUIRED PATHS ~/opencv_build/opencv/build DEFAULT_PATH  )
add_definitions(-DUSING_OPENCV)

#find the pthreads package
find_package(Threads)

#find the X11 package
find_package(X11)

#find the Approximate Nearest Neighbor Library
#find_package(ANN REQUIRED)

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
endif ( MSVC )

include_directories(
					${OPENGL_INCLUDE_DIRS}
					${GLUT_INCLUDE_DIR}
					${STIM_INCLUDE_DIRS}
          			#${ANN_INCLUDE_DIR}
					${Boost_INCLUDE_DIR}
					${OpenCV_INCLUDE_DIRS}
					)

#Assign source files to the appropriate variables
#file(GLOB SRC_CPP "*.cu")
#file(GLOB SRC_H "*.h")

#create an executable file
cuda_add_executable(netmets main.cu)

#message(${GLEW_GLEW_LIBRARY})
#set the link libraries
target_link_libraries(netmets
#						${OpenGL_LIBRARIES}
						${OPENGL_gl_LIBRARY}
						${OPENGL_glu_LIBRARY}
						${GLUT_LIBRARIES}
						${GLEW_GLEW_LIBRARY}
						${CMAKE_THREAD_LIBS_INIT}
            			#${ANN_LIBRARY}
						${X11_LIBRARIES}
						${OpenCV_LIBS}
		  )
if(MSVC)
	target_link_libraries(netmets ${GLEW_LIBRARY})
endif(MSVC)


#set up copying data files
configure_file(data/00_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/00_GT.obj @ONLY)
configure_file(data/00_T.obj ${CMAKE_CURRENT_BINARY_DIR}/00_T.obj @ONLY)
configure_file(data/01_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/01_GT.obj @ONLY)
configure_file(data/01_T.obj ${CMAKE_CURRENT_BINARY_DIR}/01_T.obj @ONLY)
configure_file(data/02_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/02_GT.obj @ONLY)
configure_file(data/02_T.obj ${CMAKE_CURRENT_BINARY_DIR}/02_T.obj @ONLY)
configure_file(data/03_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/03_GT.obj @ONLY)
configure_file(data/03_T.obj ${CMAKE_CURRENT_BINARY_DIR}/03_T.obj @ONLY)
configure_file(data/04_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/04_GT.obj @ONLY)
configure_file(data/04_Ta.obj ${CMAKE_CURRENT_BINARY_DIR}/04_Ta.obj @ONLY)
configure_file(data/04_Tb.obj ${CMAKE_CURRENT_BINARY_DIR}/04_Tb.obj @ONLY)
configure_file(data/1.swc ${CMAKE_CURRENT_BINARY_DIR}/1.swc @ONLY)
configure_file(data/2.swc ${CMAKE_CURRENT_BINARY_DIR}/2.swc @ONLY)
configure_file(data/00_GT.swc ${CMAKE_CURRENT_BINARY_DIR}/00_GT.swc @ONLY)
configure_file(data/h1_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/h1_GT.obj @ONLY)
configure_file(data/h2_GT.obj ${CMAKE_CURRENT_BINARY_DIR}/h2_GT.obj @ONLY)
configure_file(data/h1_T.obj ${CMAKE_CURRENT_BINARY_DIR}/h1_T.obj @ONLY)
configure_file(data/h2_T.obj ${CMAKE_CURRENT_BINARY_DIR}/h2_T.obj @ONLY)
