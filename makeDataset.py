from asyncore import write
import random
import string

def ramdomLine (NColum):

    line=random.choice(string.ascii_letters)
    for i in range(NColum-1):
        line=line+',' + random.choice(string.ascii_letters)

    return line

def makeCSVFile (fileName, Nrows, NColum):
    
    f = open(f"{fileName}", "w")
    for i in range(Nrows):
        f.write(ramdomLine(NColum)+"\n")

    f.close()

makeCSVFile("matrix.txt",10000,70)