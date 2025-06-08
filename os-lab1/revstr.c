#include<linux/kernel.h>
#include<linux/syscalls.h>
#include<linux/sched.h>
#include<linux/uaccess.h>

SYSCALL_DEFINE2(revstr, char __user* , msg , size_t ,len)
{
	//char kernel_buffer[256];
	//char revstr_buffer[256];
	char *kernel_buffer = kmalloc(len + 1 , GFP_KERNEL);
	char *revstr_buffer = kmalloc(len + 1, GFP_KERNEL);
	int index_head = 0;
	int index_end = len -1;
	//printk("in the revstr syscall");
	/*if(len > sizeof(kernel_buffer))
	{
		printk("error1");
		return 1;
	}*/
	if(!kernel_buffer || !revstr_buffer)
	{
		printk("error 1\n");
		return 1;
	}
	if(copy_from_user(kernel_buffer, msg , len))
	{
		printk("error2");
		return 1;
	}
	kernel_buffer[len] = '\0';
	printk("The origin string: %s\n",kernel_buffer);
	for(int i = 0 ; i < len ; i++)
	{
		//strcpy(revstr_buffer[index_end] , kernel_buffer[index_head]);
		revstr_buffer[index_end] = kernel_buffer[index_head];
		index_head++;
		index_end--;
	}
	printk("The reversed string: %s\n",revstr_buffer);
	if(copy_to_user( msg , revstr_buffer , len))
	{
		printk("error3");
		return 1;
	}
	kfree(kernel_buffer);
	kfree(revstr_buffer);
	//printk("reveived from user %s\n",kernel_buffer);
	//printk("revtsr_syscall\n");	
        return 0;
}


