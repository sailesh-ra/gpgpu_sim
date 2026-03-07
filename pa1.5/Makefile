main: main.cu
	nvcc -O3 -std=c++17 -arch=sm_89 main.cu -o main
naive: main_naive.cu
	nvcc -O3 -std=c++17 -arch=sm_89 main_naive.cu -o main_naive
tensor: main_tensor.cu
	nvcc -O3 -std=c++17 -arch=sm_89 main_tensor.cu -o main_tensor
pa1p5_testbench: launchStudentKernel.cu
	nvcc -O3 -std=c++17 -arch=sm_89 launchStudentKernel.cu -L/mnt/coe/workspace/ece/ece786-spr26 -lpa1p5_testbench -o pa1p5_testbench 
grade: 
	/mnt/coe/workspace/ece/ece786-spr26/pa1p5_grade.sh
