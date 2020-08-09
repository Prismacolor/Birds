import pandas
import csv
import tailer
import re

statelist = ['US-AL', 'US-AK', 'US-AZ', 'US-AR', 'US-CA', 'US-CO', 'US-CT', 'US-DE', 'US-FL', 'US-GA',
            'US-HI', 'US-ID', 'US-IL', 'US-IN', 'US-IA', 'US-KS', 'US-KY', 'US-LA', 'US-ME', 'US-MD',
            'US-MA', 'US-MI', 'US-MN', 'US-MS', 'US-MO', 'US-MT', 'US-NE', 'US-NV', 'US-NH', 'US-NJ',
            'US-NM', 'US-NY', 'US-NC', 'US-ND', 'US-OH', 'US-OK', 'US-OR', 'US-PA', 'US-RI', 'US-SC',
            'US-SD', 'US-TN', 'US-TX', 'US-UT', 'US-VT', 'US-VA', 'US-WA', 'US-WV', 'US-WI', 'US-WY']

# working_file = open("C:\\Data Projects\\Cali_Bird_Data\\ebd_US-CA_relMay-2020.txt", 'r+', encoding='utf-8')
# working_file_tx = open("C:\\Data Projects\\TX_Bird_Data\\ebd_US-TX_relMay-2020.txt", 'r+', encoding='utf-8')

# sample1 = tailer.head(open("C:\\Data Projects\\Cali_Bird_Data\\ebd_US-CA_relMay-2020.txt", encoding='utf-8'), 10000)
# working_sample1 = open("C:\\Data Projects\\Cali_Bird_Data\\ebd_US-CA_working_sample-2020.txt", 'r+', encoding='utf-8')

# sample_tx = tailer.head(open("C:\\Data Projects\\TX_Bird_Data\\ebd_US-TX_relMay-2020.txt", encoding='utf-8'), 20000)
# working_sample_tx = open("C:\\Data Projects\\TX_Bird_Data\\ebd_US-TX_relMay-2020_sample.txt", 'w+', encoding='utf-8')

tristate_docs = [
    "C:\\Data Projects\\Tristate_Bird_Data\\ebd_US-IN_relMay-2020.txt",
    "C:\\Data Projects\\Tristate_Bird_Data\\ebd_US-KY_relMay-2020.txt",
    "C:\\Data Projects\\Tristate_Bird_Data\\ebd_US-OH_relMay-2020.txt"
]


def convert_data(sample, working_sample):
    newline_reg = 'URN:'

    for item in sample:
        item_final = re.sub(newline_reg, '\n', item)
        working_sample.write(item_final)
        # print(item)
    working_sample.close()


# convert_data(sample1, working_sample1)
# convert_data(sample_tx, working_sample_tx)

for doc in tristate_docs:
    data_state = ''
    for state in statelist:
        if state in doc:
            print('hooray')
            data_state = state
    try:
        sample = tailer.head(open(doc, encoding='utf-8'), 15000)
    except Exception as e:
        t = 5
    working_file_name = 'C:\\Data Projects\\Tristate_Bird_Data\\' + data_state + '_data_subset.txt'
    working_file = open(working_file_name, 'w+', encoding='utf-8')

    convert_data(sample, working_file)
