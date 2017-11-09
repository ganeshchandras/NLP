import csv
import pickle
import pandas as pd
from collections import Counter
import numpy as np
import argparse


def word_tag_count(train_file):
    word_count = {}
    tag_counts = {}
    start = '<s>'
    tag_counts[start] = 0
    with open(train_file, 'r') as csvfile:
        sentence_reader = csv.reader(csvfile, delimiter='\t')
        for row in sentence_reader:
            if len(row) == 0:
                continue

            if row[1] not in word_count:
                word_count[row[1]] = 1
            else:
                word_count[row[1]] += 1

    with open(train_file, 'r') as csvfile:
        sentence_reader = csv.reader(csvfile, delimiter='\t')
        for row in sentence_reader:
            if len(row) == 0:
                tag_counts[start] += 1
                continue

            if row[2] not in tag_counts:
                tag_counts[row[2]] = 1
            else:
                tag_counts[row[2]] += 1

    return (word_count, tag_counts)


def fill_word_given_tags(word_df, tags_df,unk,train_file):
    start = '<s>'
    row_prev = start
    flag = 0
    with open(train_file, 'r') as csvfile:
        students_reader = csv.reader(csvfile, delimiter='\t')
        for row in students_reader:
            if flag == 0:
                if row[1] not in unk:
                    word_df.set_value('<s>',row[1], word_df.get_value('<s>', row[1]) + 1)
                else:
                    word_df.set_value('<s>', 'unk', word_df.get_value('<s>', 'unk') + 1)
                flag = 1
            if len(row) == 0:
                flag = 0
                row_prev = start
                continue
            if row[1] not in unk:
                word_df.set_value(row[2], row[1], word_df.get_value(row[2], row[1]) + 1)
            else:
                word_df.set_value(row[2], 'unk', word_df.get_value(row[2], 'unk') + 1)
            tags_df.set_value(row_prev,row[2],tags_df.get_value(row_prev, row[2]) + 1)
            row_prev = row[2]


def my_answer(list_of_list,test_file):
    read_file = open(test_file,'r')
    write_file = open('Satish-Ganesh-assgn2-test-output.txt','w')
    sentence_reader = csv.reader(read_file, delimiter='\t')
    string_to_write = ''
    for tags in list_of_list:
        i=0
        for row in sentence_reader:
            if len(row) == 0:
                string_to_write += '\n'
                break
            string_to_write += row[0]+'\t'+row[1]+'\t'+tags[i]+'\n'
            i += 1
    print(string_to_write)
    print("end")
    write_file.write(string_to_write)
    read_file.close()
    write_file.close()


def viterbi(word_df,tag_df,unk,test_file):
    start = 1
    word_df = word_df.drop(['<s>'])
    word_index_names = list(word_df.index)
    word_df_sum = word_df.sum(axis=1)
    tag_df_sum = tag_df.sum(axis=1)
    vit = pd.DataFrame(columns=['A'])
    answer = []
    count = []
    num_lines = 0
    line_count = 0
    with open(test_file,'r') as count_file:
        num_lines = sum(1 for line in count_file)
    with open(test_file, 'r') as csvfile:
        students_reader = csv.reader(csvfile, delimiter='\t')
        for row in students_reader:
            line_count += 1
            if len(row) == 0:
                answer = []
                col_names = list(reversed(list(vit)))
                dot = col_names.pop(0)
                list_of_tup = list(vit[dot][vit[dot].notnull()])
                max_tup = max(list_of_tup, key=lambda x: x[1])
                answer.append(max_tup[2])
                for name in col_names:
                    val = vit.get_value(answer[-1],name)[2]
                    answer.append(val)
                answer_list = list(reversed(answer))
                answer_list.pop(0)
                answer_list.append('.')
                count.append(answer_list)
                vit = pd.DataFrame(columns=['A'])
                start = 1
                continue
            if(start == 1):
                row_name = ''
                if row[1] in list(word_df):
                    row_name = row[1]
                else:
                    row_name = 'unk'
                vit = pd.DataFrame(index=word_index_names, columns=[row[0]])
                non_zero_i  = word_df[row_name].nonzero()[0]
                for val in non_zero_i:
                    tag = word_index_names[val]
                    obs_probab = word_df.get_value(tag,row_name)/word_df_sum[val]
                    transition_probab = tag_df.get_value('<s>',tag)/tag_df_sum[0]
                    product = obs_probab * transition_probab
                    vit.set_value(tag,row[0],(tag,product,'<s>'))
                    start = 0
            else:
                row_name = ''
                if row[1] in list(word_df):
                    row_name = row[1]
                else:
                    row_name = 'unk'
                non_zero_i = word_df[row_name].nonzero()[0]
                vit[row[0]] = np.nan
                vit[row[0]] = vit[row[0]].astype(object)
                for val in non_zero_i:
                    tag = word_index_names[val]
                    #print((vit.iloc[:,-2].notnull()))
                    prev_tags = vit.index[(vit.iloc[:,-2].notnull())].tolist()
                    max_val = []
                    max_list = []
                    for prev in prev_tags:
                        #print(prev,tag,row[1])
                        obs_probab = word_df.get_value(tag,row_name)/word_df_sum[val]
                        transition_probab = tag_df.get_value(prev,tag)/tag_df_sum[list(tag_df.index).index(prev)]
                        #print(vit.iloc[:,-2].loc[prev][1])
                        prev_probab = vit.iloc[:,-2].loc[prev][1]
                        product = obs_probab * transition_probab * prev_probab
                        max_val.append(product)
                        max_list.append(prev)
                    vit.set_value(tag,row[0],(tag,max(max_val),max_list[max_val.index(max(max_val))]))
    if line_count == num_lines:
        answer = []
        col_names = list(reversed(list(vit)))
        dot = col_names.pop(0)
        list_of_tup = list(vit[dot][vit[dot].notnull()])
        max_tup = max(list_of_tup, key=lambda x: x[1])
        answer.append(max_tup[2])
        for name in col_names:
            val = vit.get_value(answer[-1], name)[2]
            answer.append(val)
        answer_list = list(reversed(answer))
        answer_list.pop(0)
        # print(answer_list)
        answer_list.append('.')
        count.append(answer_list)
        return count

def baseline(word_df,tag_df,test_file):
    start = 1
    word_df = word_df.drop(['<s>'])
    answer = []
    count_of_counts = []
    num_lines = 0
    line_count = 0

    with open(test_file,'r') as count_file:
        num_lines = sum(1 for line in count_file)

    with open(test_file, 'r') as csvfile:
        students_reader = csv.reader(csvfile, delimiter='\t')
        for row in students_reader:
            line_count += 1
            if len(row) == 0:
                count_of_counts.append(answer)
                answer = []
                continue
            else:
                row_name = ''
                if row[1] in list(word_df):
                    row_name = row[1]
                else:
                    row_name = 'unk'
                answer.append(word_df[row_name].idxmax())
    if line_count == num_lines:
        count_of_counts.append(answer)
        return count_of_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_file")
    parser.add_argument("test_file")
    args = parser.parse_args()
    training_file = args.training_file
    test_file = args.test_file
    word_count, tag_count = word_tag_count(training_file)

    #print(word_count)

    unk = []
    tag_index = list(tag_count.keys())
    for key, val in word_count.items():
        if val == 1:
            unk.append(key)
    for p in unk:
        word_count.pop(p)
    word_index = list(word_count.keys())
    word_given_tag_array = np.zeros(shape=(len(tag_index), len(word_index)))
    transition_array = np.ones(shape=(len(tag_index), len(tag_index)))
    word_given_tag = pd.DataFrame(word_given_tag_array, index=tag_index, columns=word_index)
    transition_matrix = pd.DataFrame(transition_array, index=tag_index, columns=tag_index)
    word_given_tag['unk'] = pd.Series(np.zeros(len(tag_index)), index=word_given_tag.index)
    fill_word_given_tags(word_given_tag, transition_matrix,unk,training_file)
    base = baseline(word_given_tag,transition_matrix,test_file) #gives baseline answer
    print("Print Baseline")
    print(base)
    print("End Baseline")
    final = viterbi(word_given_tag,transition_matrix,unk,test_file) # gives answer by viterbi
    print("Print Viterbi")
    print(final)
    print("End Viterbi")
    my_answer(final,test_file)

if __name__ == '__main__':
    main()
