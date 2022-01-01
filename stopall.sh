ps -a|grep python|awk '{print $1}'|xargs kill -9
sleep 3
