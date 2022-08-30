import re


def replace(str):
    str = str.replace("&#xA;", "")
    str = str.replace("&apos;", "")
    str = str.replace("&amp;", "")
    str = str.replace("&#xA;", "\n")
    str = str.replace("&nbsp;", " ")
    str = str.replace("nbsp;", " ")
    str = str.replace("&lt;", "<")
    str = str.replace("&lt", "<")
    str = str.replace("lt;", "<")
    str = str.replace("?gt;", ">")
    str = str.replace("&gt;", ">")
    str = str.replace("&gt", ">")
    str = str.replace("gt;", ">")
    str = str.replace("<;code>;", "<code>")
    str = str.replace("<;/code>;", "</code>")
    str = str.replace("&amp;lt;", "<")
    str = str.replace("&amp;gt;", ">")
    str = str.replace("&quot;", "\"")
    str = str.replace("quot;", "\"")
    str = str.replace("&amp;quot;", "\"")
    str = str.replace("<br>;", "")
    str = str.replace("</br>;", "")
    str = str.replace("<;br>;", "")
    str = str.replace("<;p>;", "")
    str = str.replace("<;/p>;", "")
    str = str.replace("<;pre>;", "<ref>")
    str = str.replace("<;/pre>;", "</ref>")
    str = str.replace("<;a", "")
    str = str.replace("<;/a>;", "")
    str = str.replace("<;/a>;", "")
    str = str.replace("&amp;amp;", "&&")
    str = str.replace("amp;amp;", "&&")
    str = str.replace("&#xD;", "\n")
    str = str.replace("amp;", "&")
    str = str.replace("&reg;", "®")
    str = str.replace("reg;", "®")
    str = str.replace("&copy;", "©")
    str = str.replace("copy;", "©")

    str = str.replace("</a>", "")
    str = str.replace("</tr>", "")
    str = str.replace("</th>", "")
    str = str.replace("</td>", "")
    str = str.replace("<div>", "")
    str = str.replace("</div>", "")
    str = str.replace("<body>", "")
    str = str.replace("</dody>", "")
    str = str.replace("<html>", "")
    str = str.replace("</html>", "")
    str = str.replace("<span>", "")
    str = str.replace("</span>", "")
    str = str.replace("<li>", "")
    str = str.replace("</li>", "")
    str = str.replace("<p>", "")
    str = str.replace("</p>", "")
    str = str.replace("</option>", "")

    r1 = re.sub("<a+.*?>", "", str)
    r2 = re.sub("<t+.*?>", "", r1)
    r3 = re.sub("</t+.*?>", "", r2)
    r4 = re.sub("<input+.*?>", "", r3)
    r5 = re.sub("<img+.*?>", "", r4)
    r6 = re.sub("<select name+.*?>", "", r5)
    r7 = re.sub("<option value+.*?>", "", r6)

    return r7


def parse(str):
    index1 = 0
    final = " "
    while (True):
        start = str.find("&lt;", index1)
        if start == -1: break
        word = str[index1 + 4:start]
        final = final + word
        end = str.find("&gt;", start)
        pre = str[start + 4:end]
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

def replace_special_tokens(str):
    '''split str with whitespace'''
    res = str.split(" ")
    for t in res:
        if t == "": res.remove("")

    '''remove incorrect code tagging'''
    for i in range(len(res)):
        if res[i] == '<code>':
            for j in range(i, len(res)):
                if res[j] == '</code>':
                    if j - i <= 2:
                        res[i] = ''
                        res[j] = ''
                    break

    '''remove incorrect html tagging'''
    for i in range(len(res)):
        if res[i] == '<html>':
            for j in range(i, len(res)):
                if res[j] == '</html>':
                    if "http" not in res[i + 1]:
                        res[i] = ''
                        res[j] = ''
                    break

    '''remove incorrect table tagging'''
    for i in range(len(res)):
        if res[i] == '<table>':
            for j in range(i, len(res)):
                if res[j] == '</table>':
                    if j - i <= 4:
                        res[i] = ''
                        res[j] = ''
                    break

    for i in range(len(res)):
        if "http://" in res[i] and i >= 1 and res[i - 1] != "<html>":
            res[i] = "<html> " + res[i] + " </html>"

    '''clean whitespace'''
    for t in res:
        if t == "": res.remove("")
    context = ' '.join(res)
    return context


path = "train.txt"
outpath = "output.txt"
file = open(path, "r", encoding='utf-8')
outfile = open(outpath, "w", encoding='utf-8')
for line in file.readlines():
    str = line
    tot = tot + 1
    if tot > 5000000: break
    start = str.find("Body=")
    end = str.find("Owner")
    if start != -1 and end != -1:
        str = str[start + 6:end - 2]
        str = replace(str)
        str = parse(str)
        str = replace_special_tokens(str)
        outfile.write(str)
