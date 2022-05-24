from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
"""This script combines the files for the Regulation Room and Change My View
datasets. It generates a single .csv file containing the combined data.

NOTE: this assumes you have already ran `preprocess.py` in the regulation_room
directory.
"""

rr_train = pd.read_csv("../regulation_room/train.csv")
rr_test = pd.read_csv("../regulation_room/test.csv")
cmv_train = pd.read_csv("../change_my_view/train_set.csv")
cmv_test = pd.read_csv("../change_my_view/test_set.csv")

###################
# Regulation Room #
###################

# Combine train and test sets, and add a field indicating source
rr = pd.concat([rr_train, rr_test])
rr['source'] = 'Regulation Room'

# Drop fields that aren't in both datasets, and rename fields to a common name
rr = rr[[
    'label', 'text', 'rule_name', 'comment_number', 'proposition_number',
    'source'
]]

# Split the label into two columns
rr['verifiability'] = rr.label
rr['verifiability'] = rr['verifiability'].map({
    'u': 'unverifiable',
    'e': 'verifiable',
    'n': 'verifiable'
})

rr['experientiality'] = rr.label
rr['experientiality'] = rr['experientiality'].map({
    'u': np.nan,
    'e': True,
    'n': False
})

# Drop original label column
rr = rr.drop(columns=['label'])

# Rename columns to more general names
rr = rr.rename(
    columns={
        'text': 'sentence',
        'rule_name': 'thread_id',
        'comment_number': 'comment_id',
        'proposition_number': 'sentence_index'
    })

##################
# Change My View #
##################

# Combine train and test sets, and add a field indicating source
cmv = pd.concat([cmv_train, cmv_test])
cmv['source'] = 'Change My View'

# Drop fields that aren't in both datasets, and rename fields to a common name
cmv = cmv[[
    'sentence', 'thread_id', 'comment_id', 'id', 'verif', 'personal', 'source'
]]

# Add a sentence index
min_comment_sentence_ids = (cmv.rename(columns={
    'id': 'min_id'
}).groupby(by=['thread_id', 'comment_id'])['min_id'].min())
cmv = cmv.merge(min_comment_sentence_ids,
                on=['thread_id', 'comment_id'],
                how='inner')
cmv['sentence_index'] = cmv['id'] - cmv['min_id']
cmv = cmv.drop(columns=['id', 'min_id'])

# Rename columns to more general names
cmv = cmv.rename(columns={
    'verif': 'verifiability',
    'personal': 'experientiality'
})

# Change labels to match other data set
cmv['verifiability'] = cmv['verifiability'].map({
    'UnVerif': 'unverifiable',
    'Verif': 'verifiable',
    'NonArg': 'nonargument'
})
cmv['experientiality'] = cmv['experientiality'].map(
    {
        'NonPers': False,
        'Pers': True
    }, na_action='ignore')

#########
# Merge #
#########

# Combine data sets
merged_data = pd.concat([rr, cmv])

# Save to csv
merged_data.to_csv('./all.csv')


###################
# Generate splits #
###################

# Add stratifier column (utility)
merged_data['stratifier'] = merged_data['verifiability'].astype(
    str) + '_' + merged_data['experientiality'].astype(str)

# Remove incorrectly labeled item as there's only one occurence, which breaks
# the train_test_split function
merged_data = merged_data[merged_data['stratifier'] != 'verifiable_nan']

ratio_train = 0.7
ratio_test = 0.2
ratio_val = 0.1

# First split into test and an intermediary set
intermediary, test, _, _ = train_test_split(merged_data,
                                            merged_data['stratifier'],
                                            test_size=ratio_test,
                                            stratify=merged_data['stratifier'])

df_intermediary = pd.DataFrame(intermediary)
df_test = pd.DataFrame(test)

# Make sure ratio is correct (https://datascience.stackexchange.com/a/55322)
ratio_remaining = 1 - ratio_test
ratio_val_adjusted = ratio_val / ratio_remaining

# Next split intermediary set into train and validation
train, validation, _, _ = train_test_split(
    df_intermediary,
    df_intermediary['stratifier'],
    test_size=ratio_val_adjusted,
    stratify=df_intermediary['stratifier'])

df_train = pd.DataFrame(train)
df_validation = pd.DataFrame(validation)

# Show distributions for all sets to show successful stratification
print("==SPLIT RESULT==")
print(
    f"train.csv (n={len(df_train)}; {len(df_train) / len(merged_data) * 100:.2f}%)",
    df_train['stratifier'].value_counts() / len(df_train),
    sep='\n')
print()
print(f"test.csv (n={len(df_test)}; {len(df_test) / len(merged_data) * 100:.2f}%)",
      df_test['stratifier'].value_counts() / len(df_test),
      sep='\n')
print()
print(
    f"validation.csv (n={len(df_validation)}; {len(df_validation) / len(merged_data) * 100:.2f}%)",
    df_validation['stratifier'].value_counts() / len(df_validation),
    sep='\n')

# Save splits (without stratifier, that's just a utility column)
df_train.drop(columns=['stratifier']).to_csv('./train.csv')
df_test.drop(columns=['stratifier']).to_csv('./test.csv')
df_validation.drop(columns=['stratifier']).to_csv('./validation.csv')
