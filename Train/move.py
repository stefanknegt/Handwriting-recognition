import os

def main():
    for filename in os.listdir('data'):
        if 'Wrd_' in filename:
            directory = filename[0:12]
            if not os.path.exists('data/'+directory):
                os.makedirs('data/'+directory)
            os.rename('data/'+filename, 'data/'+directory+'/'+filename)
        else:
            directory = filename[0:4]
            if not os.path.exists('data/'+directory):
                os.makedirs('data/'+directory)
            os.rename('data/'+filename, 'data/'+directory+'/'+filename)

def count():
    txt = ''
    for dir in os.listdir('data'):
        txt = txt+dir
        count = 0
        pa = os.path.join('data', dir)
        for filename in os.listdir(pa):
            pat = os.path.join(pa, filename)
            if os.path.isfile(pat):
                count += 1
        txt = txt + ', ' + str(count) + '\n'
        
        
    text_file = open("Occurences.txt", "w")
    text_file.write(txt)
    text_file.close()

if __name__ == '__main__':
    main()