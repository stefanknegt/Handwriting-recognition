
# NOT FINISHED YET
import os

# this list will contain IoU for all bounding boxes in the lines of the original xml files and our corresponding
# updated xml file. The format is [XA1, XA2, YA1, YA2, UTFA, FILENAME, XB1, XB2, YB1, YB2, IoU] where XA1 is
# the left x coordinate of the original bounding box and XB1 the left x coordinate of our updated bounding box
with open('intersections.csv', 'w') as csv_file:
    csv_file.write("XA1, XA2, YA1, YA2, UTFA, FILENAME, XB1, XB2, YB1, YB2, IoU, \n")


for i in range(1, 3):  # 'for i range(1,13)' gets all folders
    rel_path = os.path.relpath('../data/Train/lines+xml/' + str(i) + '/')
    path = os.path.join(os.getcwd(), rel_path)
    files = os.listdir(path)
    print(path)
    j=0
    while j < len(files):  # 'for file in files' gets all files
        if '.pgm' in files[j]:
            j+=1
        else:
            lines = [line.rstrip('\n') for line in open(path+"/"+files[j])]
            x = '-x='
            y = '-y='
            w = '-w='
            h = '-h='
            utf = '<utf> '
            for line in lines:
                # Find coordinates of boundary box in original xml file and save to originalxml list
                xlo = line.find(x) + len(x)
                ylo = line.find(y) + len(y)
                hlo = line.find(h) + len(h)
                wlo = line.find(w) + len(w)
                utflo = line.find(utf) + len(utf)
                X = int(line[xlo:xlo + 4])
                Y = int(line[ylo:ylo + 4])
                W = int(line[wlo:wlo + 4])
                H = int(line[hlo:hlo + 4])
                UTF = line[utflo:utflo + 4]

                # Lines in the next xml file which is our updated xml corresponding to the original xml
                linesupdated = [lineupdated.rstrip('\n') for lineupdated in open(path+"/"+files[j+1])]
                iou = 0
                for lineupdated in linesupdated:
                # Find x coordinates of boundary boxes in updated xml file
                    try:
                        x2lo = lineupdated.find(x) + len(x)
                        w2lo = lineupdated.find(w) + len(w)
                        y2lo = lineupdated.find(y) + len(y)
                        h2lo = lineupdated.find(h) + len(h)
                        X2 = int(lineupdated[x2lo:x2lo + 4])
                        W2 = int(lineupdated[w2lo:w2lo + 4])
                        Y2 = int(lineupdated[y2lo:y2lo + 4])
                        H2 = int(lineupdated[h2lo:h2lo + 4])
                        # If there is overlap between the two bounding boxes in the x direction
                        if ((X2+W2) >= X) and ((X+W) >= X2):
                            # Save the IoU
                            overlap = max(0, min(X+W, X2+W2) - max(X, X2)) * max(0, min(Y+H, Y2+H2) - max(Y, Y2))
                            union = (W*H)+(W2*H2)-overlap
                            ioutemp = float(overlap)/float(union)
                            if ioutemp > iou:
                                iou = ioutemp
                                XB1 = X2
                                XB2 = X2+W2
                                YB1 = Y2
                                YB2 = Y2+H2
                    except ValueError:
                        continue
                with open('intersections.csv', 'a') as csv_file:
                    csv_file.write(str(X)+", "+str(X+W)+", "+str(Y)+", "+str(Y+H)+", "+UTF+", "+files[j]+", "+str(XB1)+
                                   ", "+str(XB2)+", "+str(YB1)+", "+str(YB2)+", "+str(iou)+", \n")

            j+=2




