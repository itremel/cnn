import os
import re
#das hier kann bei Python3 eigentlich raus weil das eh Standardmäßig drin ist 

#xccx88 is the dots over A,O,U --> ÄÖÜ
p = re.compile(b'\xcc\x88')

for filename in os.listdir(b'.'):
 	   m = p.search(filename)
    if(m!=None):
        print(repr(m.group()))
        print(filename)
        print(filename[7:9])
        print("whih leads to ")
        print(filename[:7]+ b"e"+filename[9:])
        os.rename(filename, filename[:7]+ b"e"+filename[9:])
