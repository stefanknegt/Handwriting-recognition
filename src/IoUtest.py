# SI= Max(0, Min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))
# then the union will be S=SA+SB-SI
# And finally, the ratio will be SI / S.

# NOT FINISHED YET

for i in range(1, 13):  # 'for i range(1,13)' gets all folders
    rel_path = os.path.relpath('../data/Train/lines+xml/' + str(i) + '/')
    path = os.path.join(os.getcwd(), rel_path)
    files = os.listdir(path)
    print(path)
    for j in range(len(files)):  # 'for file in files' gets all files
        if '.pgm' in files(j):
            continue
        else:
            lines = [line.rstrip('\n') for line in open(path)]
            x = '-x='
            y = '-y='
            w = '-w='
            h = '-h='
            utf = '<utf> '
            for line in lines:
                # Find coordinates of boundary box in original xml file
                xlo = line.find(x) + len(x)
                ylo = line.find(y) + len(y)
                hlo = line.find(h) + len(h)
                wlo = line.find(w) + len(w)
                X = int(line[xlo:xlo + 4])
                Y = int(line[ylo:ylo + 4])
                W = int(line[wlo:wlo + 4])
                H = int(line[hlo:hlo + 4])
                XA1 = X
                XA2 = X+W
                YA1 = Y
                YA2 = Y+H
                # Find coordinates of boundary box in updated xml file
                xlo = line.find(x) + len(x)
                ylo = line.find(y) + len(y)
                hlo = line.find(h) + len(h)
                wlo = line.find(w) + len(w)
                X = int(line[xlo:xlo + 4])
                Y = int(line[ylo:ylo + 4])
                W = int(line[wlo:wlo + 4])
                H = int(line[hlo:hlo + 4])
                XA1 = X
                XA2 = X+W
                YA1 = Y
                YA2 = Y+H

