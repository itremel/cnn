import os
import re
#das hier kann bei Python3 eigentlich raus weil das eh Standardmäßig drin ist 
####-*- coding: utf-8 -*-

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

#s = "Over \u0e55\u0e57 \u4500  Ö 57 flavours"
#s = b'O\xcc\x88_up_145_point_8.png'

#fn = 'filename\u4500abc'
#f = open(fn, 'w')
#f.close()

#print(os.listdir(b'.'))
#print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
#print(os.listdir('.'))