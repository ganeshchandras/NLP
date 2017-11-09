import argparse
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
def file_scan(pos_file,neg_file):
    #words_df = pd.DataFrame(columns=['Positive','Negative'])
    words = {}
    lines = open(pos_file, encoding='utf8').read().splitlines()
    #lines = open("hotelPosT-train.txt",encoding='utf8').read().splitlines()
    for line in lines:
        line = re.sub('ID-[0-9][0-9][0-9][0-9]\s','',line)
        list_line = re.findall("(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+(?=:(?!\S))", line)
        filtered_line = [word for word in list_line if word not in stopwords.words('english')]
        filtered_line = [x.lower() for x in filtered_line]
        for word in filtered_line:
            if word in words:
                words[word][0] += 1
            else:
                words[word] = [2,1]

    lines = open(neg_file, encoding='utf8').read().splitlines()
    #lines = open("hotelNegT-train.txt", encoding='utf8').read().splitlines()
    for line in lines:
        line = re.sub('ID-[0-9][0-9][0-9][0-9]\s', '', line)
        list_line = re.findall("(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+(?=:(?!\S))", line)
        filtered_line = [word for word in list_line if word not in stopwords.words('english')]
        filtered_line = [x.lower() for x in filtered_line]
        for word in filtered_line:
            if word in words:
                words[word][1] += 1
            else:
                words[word] = [1, 2]

    return words


def naive_bayes(words,test_file):
    lines = open(test_file, encoding='utf8').read().splitlines()
    #lines = open("finalTest.txt", encoding='utf8').read().splitlines()
    answrite = open("answer.txt",'w',encoding='utf8')
    line_count = 0
    pos_count = 0
    neg_count = 0
    stopWord = set(stopwords.words('english'))
    stopWord.update(
        ['','internet','high', 'speed', 'thing', '--', 'place', 'thought', 'paid', 'overall', 'i\'m', 'i\'ll', 'house', 'carrying',
         'around', 'many', 'staying', 'stayed', 'enjoy', 'without', 'partying', 'sitting', 'coming', 'provided',
          'entered', 'events', 'going', 'city', 'life', 'sheraton',  'strip','location',  'questions', 'passed', 'food', 'store', 'actually', 'fan', 'tv',
         'pool', 'paint', 'walls', 'restaurant', 'downstairs','cigarette', 'indoor','street', 'union', 'entire', 'hear', 'wifi','much', 'came', 'late', 'last', 'spot', 'buses', 'ran', 'checked', 'checking',
         'clerk', 'two', '&', 'found', 'king', 'suite', 'called', 'motel', 'back', 'front', 'rooms',
         'breakfast', 'continental', 'area', 'staff', 'located', 'though',  'first', 'ac','us', 'casino', 'starter', 'get', 'hotels', 'movie', 'room'])
    stop_list = list(stopWord)
    for line in lines:
        line_count += 1
        id_no = re.match('ID-[0-9][0-9][0-9][0-9]\s', line)
        line = re.sub('ID-[0-9][0-9][0-9][0-9]\s', '', line)
        list_line = re.findall("(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+(?=:(?!\S))", line)
        filtered_line = [word for word in list_line if word not in stop_list]
        filtered_line = [x.lower() for x in filtered_line]
        pos = 1
        neg = 1
        for word in filtered_line:
            if word in words:
                pos = pos * words[word][0]
                neg = neg * words[word][1]
        if pos >= neg:
            pos_count += 1
            answrite.write(id_no.group(0) + ' POS')
        else:
            neg_count += 1
            answrite.write(id_no.group(0) + ' NEG')
        answrite.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_file")
    parser.add_argument("neg_file")
    parser.add_argument("testfile")
    args = parser.parse_args()
    pos_file = args.pos_file
    neg_file = args.neg_file
    test_file = args.testfile
    words_dict = file_scan(pos_file,neg_file)
    naive_bayes(words_dict,test_file)
if __name__ == '__main__':
    main()