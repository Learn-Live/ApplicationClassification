import os
L = os.listdir( "./" )
f = open("statistics.txt","w") 
f.close()
f = open("statistics.txt","a") 

for folder in L:
	if (".py" not in folder and  ".txt" not in folder):
		num = len(os.listdir("./"+folder+"/"))
		f.write(folder+"\t"+str(num)+"\n")
	
f.close()
