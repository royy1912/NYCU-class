cmd_/home/Roy/os_lab3/Module.symvers :=  sed 's/ko$$/o/'  /home/Roy/os_lab3/modules.order | scripts/mod/modpost -m -a    -o /home/Roy/os_lab3/Module.symvers -e -i Module.symvers -T - 
