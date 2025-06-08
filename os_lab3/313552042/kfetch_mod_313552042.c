#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/device.h>
#include <linux/sysinfo.h>
#include <linux/utsname.h>
#include "kfetch.h"
#include <linux/sched/signal.h>
#include <linux/timekeeping.h>
#include <linux/mm.h>
#include <linux/swap.h>
#include <asm/processor.h>
#include <linux/smp.h>
#include <linux/cpumask.h>
#include <linux/mutex.h>
#include <linux/string.h>
static int device_number;
static struct class* kfetch_class = NULL;
static struct device* kfetch_device = NULL;
static int mask_info;
char kernel_buffer[KFETCH_BUF_SIZE];
size_t offset = 0;
int written = 0;
static DEFINE_MUTEX(my_mutex);

static void penguin(int count)
{
	switch(count)
	{
		case 0 :
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "                   ");
			offset += written;
			break;
		case 1:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n        .-.        ");
			offset += written;
			break;
		case 2:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n       (.. |       ");
			offset += written;
			break;
		case 3:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n       <>  |       ");
                        offset += written;
                        break;
		case 4:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n      / --- \\      ");
                        offset += written;
                        break;
		case 5:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n     ( |   | )     ");
                        offset += written;
                        break;
		case 6:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n   |\\\\_)__(_//|   ");
                        offset += written;
                        break;
		case 7:
			written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n  <__)------(__>  ");
                        offset += written;
                        break;

	}
}

static ssize_t kfetch_read(struct file *file_ptr , char __user *buffer , size_t len , loff_t *position)
{
	//if(!mutex_trylock(&my_mutex))
	//	return -EBUSY;
	//size_t offset = 0;	
	offset = 0;
	//int written = 0;
	written = 0;
	int line_len = 0;
	int count = 0;
	penguin(count);
	count++;
	written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "%s" , init_uts_ns.name.nodename);
	offset += written;
	penguin(count);
	count++;
	for(int i = 0 ; i < strlen(init_uts_ns.name.nodename) ; i++)
	{
		snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "-");
		offset += 1;
	}
	printk(KERN_INFO "host_name_len = %d" , sizeof(init_uts_ns.name.nodename));

	if((mask_info & KFETCH_RELEASE) != 0)
        {
                //snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n");
		//offset += 1;
		penguin(count);
		count++;
		written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "kernel:	%s" , init_uts_ns.name.release);
		offset += written;
        }
	if((mask_info & KFETCH_CPU_MODEL) != 0)
        {
                //snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n");
		//offset += 1;
		penguin(count);
		count++;
		written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "CPU:	        %s" , boot_cpu_data.x86_model_id);
		offset += written;
        }
	if((mask_info & KFETCH_NUM_CPUS) != 0)
        {
                int online_cpus = num_online_cpus();
		int total_cpus = num_possible_cpus();
		//snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n");
                //offset += 1;
		penguin(count);
		count++;
		printk(KERN_INFO "online: %d" , online_cpus);
		printk(KERN_INFO "total: %d" , total_cpus);
		written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "CPUs:	%d / %d" , online_cpus , total_cpus);
		offset += written;

        }
	if((mask_info & KFETCH_MEM) != 0)
        {
                struct sysinfo mem_info;
                unsigned long free_mem , total_mem;
                si_meminfo(&mem_info);

                total_mem = (mem_info.totalram * mem_info.mem_unit) / (1024 * 1024);
                free_mem = (mem_info.freeram * mem_info.mem_unit) / (1024 * 1024);
                //snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n");
                //offset += 1;
		penguin(count);
		count++;
                written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "Mem:	        %lu MB / %lu MB" , free_mem , total_mem);
                offset += written;
        }
	if((mask_info & KFETCH_NUM_PROCS) != 0)
        {
                struct task_struct *task;
                int process_count = 0;
                for_each_process(task)
                {
                        process_count++;
                }
                //snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n");
                //offset += 1;
		penguin(count);
		count++;
                written = snprintf(kernel_buffer + offset  , KFETCH_BUF_SIZE - offset  , "Procs:	%d",process_count);
                offset += written;
        }
	if((mask_info & KFETCH_UPTIME) != 0)
        {
                struct timespec64 uptime;
                unsigned long uptime_in_minutes;
                ktime_get_boottime_ts64(&uptime);
                uptime_in_minutes = uptime.tv_sec / 60;
                //snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "\n");
                //offset += 1;
		penguin(count);
		count++;
                written = snprintf(kernel_buffer + offset , KFETCH_BUF_SIZE - offset , "Uptime:	%lu mins",uptime_in_minutes);
                offset += written;
        }
	while(count <= 7)
	{
		penguin(count);
		count++;
	}
	if(copy_to_user(buffer , kernel_buffer , offset))
	{
		printk(KERN_INFO "copy fail");
	}
	else
	{
		printk(KERN_INFO "copy success");
	}

	//mutex_unlock(&my_mutex);

	return offset;
}
static ssize_t kfetch_write(struct file *file_ptr , const char __user *buffer , size_t len , loff_t *position)
{
	//if(!mutex_trylock(&my_mutex))
	//	return -EBUSY;


	if(copy_from_user(&mask_info , buffer , len))
	{
		printk(KERN_ALERT "Copy from user failed\n");
		return 0;
	}
	printk(KERN_INFO "Value:%d\n" , mask_info);

	//mutex_unlock(&my_mutex);

        return 0;
}
static int kfetch_open(struct inode *inodep , struct file *file_ptr)
{
	printk(KERN_INFO "Device opened\n");
	if(!mutex_trylock(&my_mutex))
		return -EBUSY;
	return 0;
}

static int kfetch_release(struct inode *inodep , struct file *file_ptr)
{
	printk(KERN_INFO "Device closed\n");
	mutex_unlock(&my_mutex);
	return 0;
}


const static struct file_operations kfetch_ops = {
        .owner          = THIS_MODULE,
        .read           = kfetch_read,
        .write          = kfetch_write,
        .open           = kfetch_open,
        .release        = kfetch_release,
};

static int __init kfetch_init(void)
{
	device_number = register_chrdev(0 , KFETCH_DEV_NAME , &kfetch_ops);
	if(device_number < 0)
	{
		printk(KERN_ALERT "Failed to build device\n");
		return device_number;
	}
	printk(KERN_INFO "Get the device number\n" , device_number);
	kfetch_class = class_create(THIS_MODULE,"kfetch_class");
	if(IS_ERR(kfetch_class))
	{
		unregister_chrdev(device_number , KFETCH_DEV_NAME);
		printk(KERN_ALERT "Fail to create class\n");
		return PTR_ERR(kfetch_class);
	}
	printk(KERN_INFO "Class create successfly\n");

	kfetch_device = device_create(kfetch_class , NULL , MKDEV(device_number,0) , NULL , KFETCH_DEV_NAME);
	if(IS_ERR(kfetch_device))
	{
		class_destroy(kfetch_class);
		unregister_chrdev(device_number , KFETCH_DEV_NAME);
		printk(KERN_ALERT "Fail to create device\n");
		return PTR_ERR(kfetch_device);
	}
	printk(KERN_INFO "Create device successfly\n");
	return 0;
}

static void __exit kfetch_exit(void)
{
	device_destroy(kfetch_class , MKDEV(device_number , 0));
	class_unregister(kfetch_class);
	class_destroy(kfetch_class);
	unregister_chrdev(device_number , KFETCH_DEV_NAME);
	printk(KERN_INFO "Device unloaded\n");
}

module_init(kfetch_init);
module_exit(kfetch_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Roy");
MODULE_DESCRIPTION("hw3");
