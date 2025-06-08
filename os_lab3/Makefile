obj-m += kfetch_mod.o
kfetch_mod-objs := kfetch_mod_313552042.o
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

load:
	sudo insmod kfetch_mod.ko

unload:
	sudo rmmod kfetch_mod
