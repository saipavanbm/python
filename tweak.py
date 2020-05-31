import os
import time
it=1
while it<=4:
	it+=1
	acc = os.popen("cat /root/pyth/accuracy.txt")
	acc1 = acc.read()
	print(acc1)
	acc2 = acc1.rstrip()
	print(acc2)
	acc3 = float(acc2)


	if acc3<80:
	    x = os.popen("cat /root/pyth/keras.py | grep model.add | wc -l")
	    x1 = x.read()
	    x2 = x1.rstrip()
	    x3 = int(x2)
	    print(x3)
	    if x3==2:
	        y = 'model.add(Dense(units=32, activation=\"relu\"))'
	    elif x3==3:
	        y = 'model.add(Dense(units=16, activation=\"relu\"))'
	    elif x3==4:
	        y = 'model.add(Dense(units=8, activation=\"relu\"))'
	    else:
	        print("Limit Reached")
	        exit()
	    os.system("sed -i '/softmax/ i {}' /root/pyth/keras.py".format(y))
	    os.system("curl -u admin:redhat http://192.168.99.101:8080/job/mlt-job2/build?token=tweak")
	    #time.sleep(45)
	    acc = os.popen("cat /root/pyth/accuracy.txt")
	    acc1 = acc.read()
	    print(acc1)
	    acc2 = acc1.rstrip()
	    print(acc2)
	    acc3 = float(acc2)
	else:
	    print("ACCURACY ABOVE 80")