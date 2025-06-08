#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif


static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x88db9f48, "__check_object_size" },
	{ 0xec37920b, "__class_create" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0xc60d0620, "__num_online_cpus" },
	{ 0x656e4a6e, "snprintf" },
	{ 0x21ea5251, "__bitmap_weight" },
	{ 0x3edbf426, "class_destroy" },
	{ 0xcbd4898c, "fortify_panic" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x92997ed8, "_printk" },
	{ 0xd0da656b, "__stack_chk_fail" },
	{ 0x7682ba4e, "__copy_overflow" },
	{ 0xa916b694, "strnlen" },
	{ 0x68ddb0dc, "init_task" },
	{ 0xe5208530, "init_uts_ns" },
	{ 0xece70a6c, "device_create" },
	{ 0xd506a70e, "class_unregister" },
	{ 0x65929cae, "ns_to_timespec64" },
	{ 0x9e683f75, "__cpu_possible_mask" },
	{ 0x5b8239ca, "__x86_return_thunk" },
	{ 0x17de3d5, "nr_cpu_ids" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0xbb9ed3bf, "mutex_trylock" },
	{ 0x40c7247c, "si_meminfo" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0x80d38716, "__register_chrdev" },
	{ 0xe61f2f6, "device_destroy" },
	{ 0xf5b00aab, "boot_cpu_data" },
	{ 0xc4f0da12, "ktime_get_with_offset" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0x4963cf87, "module_layout" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "E39A15F878CFF6BC4716083");
