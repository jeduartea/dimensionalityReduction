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

Nrows = 2560
NColum = 10
makeCSVFile("1sequence/matrix.txt",Nrows=Nrows,NColum=NColum)
#makeCSVFile("./2openMP/matrix.txt",Nrows=Nrows,NColum=NColum)

print(f"Created a test file with:\n rows: {Nrows} \n columns: {NColum}")