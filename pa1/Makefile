main: main.cu launchStudentKernel.cu // remove launchStudentKernel before submission
	nvcc -O3 -std=c++17 -arch=sm_89 main.cu -o main
pa1_testbench: launchStudentKernel.cu
	nvcc -O3 -std=c++17 -arch=sm_89 launchStudentKernel.cu -L/mnt/coe/workspace/ece/ece786-spr26 -lpa1_testbench -o pa1_testbench 
grade: 
	/mnt/coe/workspace/ece/ece786-spr26/pa1_grade.sh
