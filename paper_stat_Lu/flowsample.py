import re

s = ""
with open("flowsample.txt", "r") as rfile:
    s = rfile.read()
li = s.split("\n")
dic = {}
for line in li:
    if "SSL." in line:
        match = re.findall("SSL.([\s\S]*?)]", line)[0]
        if match in dic:
            dic[match] = dic[match] + 1
        else:
            dic.update({match: 1})
print(dic)

sum = 0
for key, value in dic.items():
    sum += value
print(sum)
'''
'{'Apple': 1, 'GoogleMaps': 9, 'Twitter': 99, 'Microsoft': 34, 'Telegram': 1, 'Skype': 7, 'Dropbox': 3, 'GooglePlus': 1, 'Office365': 4, 'YouTube': 2, 'Github': 10, 'GoogleDocs': 5, 'MS_OneDrive': 16, 'Facebook': 72, 'Yahoo': 12,
'GoogleServices': 57, 'Cloudflare': 69, 'Amazon': 371, 'Signal': 6, 'MSN': 3, 'WindowsUpdate': 16, 'Slack': 13, 'Google': 586}'
'''

'''
74
'''
26 % .14
