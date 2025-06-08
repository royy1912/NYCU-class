compiler :=g++

.PHONY:default clean

all: sched_demo_313552042

sched_demo_313552042: sched_demo_313552042.cpp
	$(compiler) -D_GNU_SOURCE sched_demo_313552042.cpp -o sched_demo_313552042 -pthread -lrt

test:
	sudo ./sched_test.sh ./sched_demo ./sched_demo_313552042
clean:
	rm sched_demo_313552042
check_1:
	sudo ./sched_demo -n 1 -t 0.5 -s NORMAL -p -1 > demo_output.txt
	sudo ./sched_demo_313552042 -n 1 -t 0.5 -s NORMAL -p -1 > my_output.txt
check_2:
	sudo ./sched_demo -n 2 -t 0.5 -s FIFO,FIFO -p 10,20 > demo_output.txt
	sudo ./sched_demo_313552042 -n 2 -t 0.5 -s FIFO,FIFO -p 10,20 > my_output.txt
check_3:
	sudo ./sched_demo -n 3 -t 1.0 -s NORMAL,FIFO,FIFO -p -1,10,30 > demo_output.txt
	sudo ./sched_demo_313552042 -n 3 -t 1.0 -s NORMAL,FIFO,FIFO -p -1,10,30 > my_output.txt


