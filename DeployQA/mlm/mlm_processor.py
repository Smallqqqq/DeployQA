import os


path = "./mlm_evalnew" #文件夹目录
output = "。/valid.txt"
out = open(output,"w")
qa_set = []
for root, dirs, files in os.walk(path):
    for file in files:
        str1 = os.path.join(root,file)
        file = open(str1,"r")
        s = file.read()
        w = out.write(s)
        w = out.write('\n')
