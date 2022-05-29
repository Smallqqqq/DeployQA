def replace(str):
    str=str.replace("&#xA;","")
    str=str.replace("&apos;","")
    str=str.replace("&amp;","")
    return str
def parse(str):
    tmp= ""
    flag = 0
    word = ""
    res = ""
    finish = 0
    index1 = 0
    final = " "
    while (True):
        start= str.find("&lt;",index1)
        if start == -1: break
        word = str[index1+4:start]
        final = final + word
        end = str.find("&gt;",start)
        pre = str[start+4:end]
        if pre == "code":
            final = final + "<code>"
        if pre == "/code":
            final = final + "</code>"
        if "href" in pre:
            final = final + "<html>"
        if pre == "/a":
            final = final + "</html>"
        if pre == "/p":
            final = final + "\n"
        index1 = end
    return final

path = "pretraintxt.txt"
outpath = "output.txt"
file = open(path,"r",encoding='utf-8')
outfile = open(outpath,"w",encoding='utf-8')
tot = 0
for line in file.readlines():
    str = line
    tot = tot + 1
    if tot > 3000000: break
    start = str.find("Body=")
    end = str.find("Owner")
    if start != -1 and end != -1:
        str = str[start + 6:end - 2]
        str = replace(str)
        str = parse(str)
        outfile.write(str)