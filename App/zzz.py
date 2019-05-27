def nr():
    nr0, nr1, nr2, nr3, nr4, nr5, nr6, nr7, nr8, nr9 = 0, 0, -2, -1, 0, -3, 0, -2, 0, -3    #2|-2 3|-1 5|-3 7|-2 9|-3
    
    for line in lines:
        line = line.strip()
        if int(line) == 0:
            nr0+=1
        elif int(line) == 1:
            nr1+=1
        elif int(line) == 2:
            nr2+=1
        elif int(line) == 3:
            nr3+=1
        elif int(line) == 4:
            nr4+=1
        elif int(line) == 5:
            nr5+=1
        elif int(line) == 6:
            nr6+=1
        elif int(line) == 7:
            nr7+=1
        elif int(line) == 8:
            nr8+=1
        elif int(line) == 9:
            nr9+=1
            
    print('0:{}\n1:{}\n2:{}\n3:{}\n4:{}\n5:{}\n6:{}\n7:{}\n8:{}\n9:{}'
          .format(nr0, nr1, nr2, nr3, nr4, nr5, nr6, nr7, nr8, nr9))

with open('./database/values.txt') as f:
    lines = f.readlines()
    
nr()
print(len(lines))
