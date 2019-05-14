import sys, os, subprocess
from subprocess import Popen
from subprocess import check_output

old10 = {0:'SSL.Amazon',1:'SSL.Cloudflare',2:'SSL.Facebook',3:'SSL.Github',4:'SSL.Google',5:'SSL.Microsoft',6:'SSL.Skype',7:'SSL.Twitter',8:'SSL.YouTube',9:'SSL.Slack'}
new10 = {0:'google',1:'twitter',2:'outlook',3:'imgur',4:'youtube',5:'cloudflare',6:'github',7:'facebook',8:'slack',9:'bing'}
newkey10 = {0:['google'],1:['twitter','twimg'],2:['live'],3:['imgur'],4:['youtube'],5:['cloudflare'],6:['github'],7:['facebook'],8:['slack'],9:['bing']}
keymap = {0:[],1:['cloudflare','imgur'],2:['facebook'],3:['github'],4:['google'],5:['live','bing'],6:[],7:['twitter','twimg'],8:['youtube'],9:['slack']}
keytoappmap = {'cloudflare': 5, 'imgur': 3, 'facebook': 7, 'github': 6, 'google': 0, 'live': 2, 'bing': 9,'twitter':1, 'twimg':1,'youtube':4,'slack':8}
res = []
start = 9
print("start:", start)
for i in range(start, start + 1):
	dicname = old10[i]
	for fileName in os.listdir("./"+dicname):
		keylist = keymap[i]
		for item in keylist:
			cmd = '\"C:\Program Files\Wireshark\\tshark.exe\" -r '+'./'+dicname+'/'+fileName+' ssl.handshake.extensions_server_name contains \"'+item+'\"'

			ret = check_output(cmd, shell=True)

			if(len(ret) > 0): # contain
				
				res.append(dicname +"\t"+ fileName +"\t"+item + '\t' +str(keytoappmap[item]))
				print(len(res))
			#output,error  = subprocess.Popen(cmd, universal_newlines=True, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
			#print(output)
with open("res"+str(start)+".txt","w") as wfile:
	wfile.write("{0}".format("\n".join(res)))