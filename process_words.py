

file = open('synset.txt', 'r')
out = open('out.txt', 'w')

i = 0
for line in file:
    i += 1
    if i > 1000:
        line = line.strip()
        line = line[3:]
        line = line.split(' ')[0]
        out.write(line + '\n')
file.close()
out.close()