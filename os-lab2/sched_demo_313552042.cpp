//#define _GNU_SOURCE
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sched.h>
#include <pthread.h>
#include <chrono>
#include <thread>
pthread_barrier_t barrier;
int  wait_time = 0;
void *thread_func(void *arg)
{
	int thread_id = *static_cast<int*>(arg);
	int policy;
	sched_param thread_param;
	pthread_getschedparam(pthread_self(),&policy,&thread_param);
	//std::cout << "thread " << thread_id << " policy is ";
	//if(policy == SCHED_OTHER) std::cout << "SCHED_NORMAL";
	//else if(policy == SCHED_FIFO) std::cout << "SCHED_FIFO";
	//std::cout << ", priority: " << thread_param.sched_priority << std::endl;
	//std::cout << "thread "<< thread_id << " reach the barrier" << std::endl;
	pthread_barrier_wait(&barrier);
	//std::cout <<"hello thread "<< thread_id <<std::endl;
	/*cpu_set_t get_cpuset;
	CPU_ZERO(&get_cpuset);
	pthread_getaffinity_np(pthread_self() , sizeof(cpu_set_t) , &get_cpuset);
	std::cout << "thread" << thread_id << "run on ";
	for(int i = 0 ; i < CPU_SETSIZE ; i++)
	{
		if(CPU_ISSET(i , &get_cpuset))
		{
			std::cout << i << std::endl;
		}
	}*/	
	for(int i = 0 ; i < 3 ; i++)
	{
		std::cout << "Thread " << thread_id << " is starting" << std::endl;
		auto start_time = std::chrono::steady_clock::now();
		auto end_time = start_time + std::chrono::milliseconds(wait_time);
		//std::cout << wait_time << std::endl;
		while(std::chrono::steady_clock::now() < end_time)
		{
		//busy waiting
		}
	}

	return NULL;

}
int main(int argc , char *argv[])
{
	cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(1 , &cpuset);
	std::cout.setf(std::ios::unitbuf);
	/*if(sched_setaffinity(0,sizeof(cpu_set_t) , &cpuset) != 0)
	{
		std::cerr <<" error set main thread" << std::endl;
		return 0;
	}*/
	int opt;
	int n = 0;
	float t = 0;
	std::vector<std::string> s_value;
	std::vector<int> p_value;
	//pthread_attr_t attr;
	sched_param param;
	while((opt = getopt(argc , argv , "n:t:s:p:")) != -1)
	{
		//std::cout << opt << std::endl;
		switch(opt)
		{
			case 'n':
			{
				n = std::stoi(optarg);
				//std::cout << "option -n with " << n << std::endl;
				break;
			}
			case 't':
			{
				t = std::stof(optarg);
				wait_time = t * 1000;
				//std::cout << "option -t with " << t << std::endl;
				break;
			}
			case 's':
			{
				std::string s_arg = optarg;
				std::stringstream s_totalstring(s_arg);
				std::string s_item;
				while(std::getline(s_totalstring , s_item , ','))
				{
					s_value.push_back(s_item);
				}
				/*std::cout << "option -s with " << std::endl;
				for(auto &val : s_value)
				{
					std::cout << " " << val;
				}
				std::cout << std::endl;*/
				break;
			}
			case 'p':
			{
				std::string p_arg = optarg;
				std::stringstream p_totalstring(p_arg); 
				std::string p_item;
				while(std::getline(p_totalstring , p_item , ','))
				{
					p_value.push_back(std::stoi(p_item));
				}
				/*std::cout << "option -p with " << std::endl;
				for(int i = 0 ; i < n ; i++)
					std::cout << p_value[i] << " ";*/
				/*for(auto val : p_value)
					std::cout << val << " ";*/
				//std::cout << std::endl;
				break;
			}
			default:
			{
				std::cerr << "input error" << std::endl;
				break;
			}
		}
	}
	std::vector<pthread_t> threads(n);
	std::vector<int> thread_id(n);
	std::vector<pthread_attr_t> attr(n);
	pthread_barrier_init(&barrier , NULL , n + 1);
	int ret= 0;
	for(int i = 0 ; i < n ; i++)
	{
		thread_id[i] = i;
		pthread_attr_init(&attr[i]);
		//pthread_attr_setstacksize(&attr ,1024 *1024);
		pthread_attr_setinheritsched(&attr[i] , PTHREAD_EXPLICIT_SCHED);
		if(s_value[i] == "NORMAL")
		{
			ret = pthread_attr_setschedpolicy(&attr[i] , SCHED_OTHER);
			if(ret != 0)
				std::cerr <<"error at set SCHED_OTHER policy at thread "<< thread_id[i] << std::endl;
			param.sched_priority = 0;
			ret = pthread_attr_setschedparam(&attr[i] , &param);
			if(ret != 0)
				std::cerr <<"error at setting SCHRD_OTHER priority at thread " << thread_id[i] << std::endl;
			//std::cout << "NORMAL" << std::endl;
		}
		else if(s_value[i] == "FIFO")
		{
			ret = pthread_attr_setschedpolicy(&attr[i] , SCHED_FIFO);
			if(ret != 0)
                                std::cerr <<"error at set SCHED_FIFO policy at thread "<< thread_id[i] << std::endl;
			param.sched_priority = p_value[i];
			ret = pthread_attr_setschedparam(&attr[i] , &param);
			if(ret != 0)
                                std::cerr <<"error at setting SCHRD_FIFO priority at thread " << thread_id[i] << std::endl;
			//int policy;
			//pthread_attr_getschedpolicy(&attr,&policy);
			//if(policy == SCHED_FIFO) std::cout << "attr is FIFO";
			//sched_param get_param;
			//pthread_attr_getschedparam(&attr,&get_param);
			//std::cout << ",priority = " << get_param.sched_priority << std::endl;
			//std::cout << "FIFO" << std::endl;
		}	
		//int policy;
                //pthread_attr_getschedpolicy(&attr[i],&policy);
                //if(policy == SCHED_FIFO) std::cout << "attr[i] is FIFO";
                //sched_param get_param;
                //pthread_attr_getschedparam(&attr[i],&get_param);
                //std::cout << ",priority = " << get_param.sched_priority << std::endl;
		if(pthread_create(&threads[i] , &attr[i] , thread_func , &thread_id[i]) != 0)
		{
			std::cerr <<"thread create fail" << std::endl;
			return 1;
		}	
		if(pthread_setaffinity_np(threads[i] , sizeof(cpu_set_t) , &cpuset) != 0)
			std::cerr <<"error setting affinity" << i+1 << std::endl;
		pthread_attr_destroy(&attr[i]);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	pthread_barrier_wait(&barrier);
	for(int i = 0 ; i < n ; i++)
		pthread_join(threads[i], NULL);
	//pthread_attr_destroy(&attr);
	pthread_barrier_destroy(&barrier);
	//std::cout <<"Main thread finish" << std::endl;
	//std::cout <<"pthread create error" << std::endl;
	return 0;
}
