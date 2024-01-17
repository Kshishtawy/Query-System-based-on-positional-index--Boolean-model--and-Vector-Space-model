import os
from natsort import natsorted
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import math
from math import log10
from tabulate import tabulate 
import numpy as np
# nltk.download('punkt')

def apply_tokenization_and_stemming(text):
    # initializing the stemmer. in our case it is the PorterStemmer
    stemmer = PorterStemmer()

    # tokenizing words
    tokenization = word_tokenize(text)

    # declearing the list of stemmed words
    stemmed_words = []

    # looping over the tokenzied words and applying stemming then appending each stemmed word to 'stemmed_words'
    for word in tokenization:
        stemmed_words.append(stemmer.stem(word))

    # returning a list of words that are tokenized and stemmed
    return stemmed_words


 #########################################################################
#   Change the "files_path" variable to your collection of .txt files   

files_path = "CHANGE ME"

 #########################################################################

document_collection = natsorted(os.listdir(files_path))
list_of_terms = []

for doc in document_collection:
    with open(f'{files_path}\{doc}', 'r') as f:
            document = f.read()
    list_of_terms.append(apply_tokenization_and_stemming(document))

# Starting second part (1) positional index

positional_index = {}

for doc_id, terms in enumerate(list_of_terms, start=1):
    for position, term in enumerate(terms, start=1):
        # if the term doesn't exist in our positional index we add it
        if term not in positional_index:
              positional_index[term] = {
                   'doc_count': 1,
                   'docs': {doc_id: [position]}
              }
        # if the term already exists we do the following
        else:
            # first, add to the count of that term
            positional_index[term]['doc_count'] += 1
            # if the doc_id is new, this means we have a new document that contain the same word
            # we add a the doc_id to our positional index along with the position of that word in that doc_id
            if doc_id not in positional_index[term]['docs']:
                positional_index[term]['docs'][doc_id] = [position]
            # if the doc_id already exists, then we add the position of that term within that doc_id
            else:
                positional_index[term]['docs'][doc_id].append(position)

# tf

term_frequency = {}

for term, info in positional_index.items():
    term_frequency[term] = {}
    for doc_id, positions in info['docs'].items():
         term_frequency[term][f"Doc{doc_id}"] = len(positions)

# converting our tf table to a pandas dataframe so we can display it
tf_table = pd.DataFrame(term_frequency).transpose().fillna(0).astype(int)
# sorting columns
tf_table = tf_table[natsorted(tf_table.columns)]


# w_tf

w_tf = tf_table.copy()

for c_name, c_value in w_tf.items():
    # print(f"Column: {c_value}")
    for index, value in c_value.items():
        if w_tf.loc[index, c_name] != 0:
            w_tf.loc[index, c_name] = (1 + log10(w_tf.loc[index, c_name]))

# Computing IDF
df_idf_table = pd.DataFrame(columns=['df', 'idf'])

N = len(document_collection)

for i, term in enumerate(positional_index):
    df_idf_table.loc[i, 'df'] = tf_table.loc[term].sum(axis=0)
    df_idf_table.loc[i, 'idf'] = log10(N/df_idf_table.loc[i, 'df'])
    
df_idf_table.index = tf_table.index

# Computing TF-IDF
tf_idf_table = tf_table.copy()

for i in range(1, tf_table.shape[1] + 1):
    tf_idf_table[f'Doc{i}'] *= df_idf_table['idf'].values

# Normalized length of docs
doc_length_index = [f"{col} length" for col in tf_idf_table.columns]

doc_length = pd.DataFrame(columns=["Euclidean length"], index=doc_length_index)

for c_name, c_value in tf_idf_table.items():
    doc_length.loc[f"{c_name} length", "Euclidean length"] = math.sqrt((c_value ** 2).sum())

# Normalized tf.idf
normalized_tf_idf = tf_idf_table.copy()

for c_name, c_value in normalized_tf_idf.items():
    doc_length_value = doc_length.loc[f"{c_name} length", "Euclidean length"]
    normalized_tf_idf[c_name] = normalized_tf_idf[c_name] / doc_length_value

# Phrase query
def intersect(postings):
    # check if postings list is empty
    if len(postings) == 0:
        return 0
    else:
        intersection_set = set(postings[0])

        for posting in postings[1:]:
            intersection_set = intersection_set.intersection(posting)
            

        return natsorted(intersection_set)

def matched_docs(query):
    query_terms = query
    postings_list = []
    for term in query_terms:
        if term in positional_index:
            postings_list.append(list(positional_index[term]['docs'].keys()))
        else:
            print("No documents found for selected query terms")
            return 0
            
    # ordering the postings list based on df (this will help with intersection)
    ordered_postings = sorted(postings_list, key=lambda item: len(item))
    
    # print(ordered_postings)
    
    # computed matched docs with intersect function on the ordered postings
    matched_docs = intersect(ordered_postings)
    
    # after looping the query_terms if no terms are matched with positional index
    # the postings_list will be empty and return 0
    if not (bool(matched_docs)):
        return 0
        
    # else we intersect term's postings lists
    else:
        return matched_docs
    
doc_id_positions = {key: value['docs'] for key, value in positional_index.items()}

def process_query(query):
    stemmed_query = apply_tokenization_and_stemming(query)
    operator = ""
    switch = 0
    phrase_one = []
    phrase_two = []
    
    for term in stemmed_query:
        if switch == 0:
            phrase_one.append(term)
        elif switch == 1:
            phrase_two.append(term)

        if term == "and":
            switch = 1
            phrase_one.pop()
            operator = "and"

        elif term == "or":
            switch = 1
            phrase_one.pop()
            operator = "or"

        elif term == "not":
            switch = 1
            phrase_one.pop()
            operator = "not"
    return phrase_one, operator, phrase_two

def phrase_query(phrase):
    
    list_of_docs = matched_docs(phrase)
    
    # matched docs for the phrase
    if len(phrase) == 1 or len(phrase) == 0:
        return list_of_docs
        
    else:
        if list_of_docs == 0:
            return 0
        else:
            
            #print(list_of_docs)
            postings = []

            # get the positions of matched docs for each term in query
            for term in phrase:
                postings_term = []
                for i in list_of_docs:
                    # print(doc_id_positions[term][i])
                    postings_term.append(int(doc_id_positions[term][i][0]))
                postings.append(postings_term)

            #print(postings)

            # using the 666 method
            # postings will have same values for a successfully detected phrase
            for index, list_of_positions in enumerate(postings):
                for pos_idx, position in enumerate(list_of_positions):
                    postings[index][pos_idx] = position + (len(postings) - index)


            final_list = []
            for index, list_of_positions in enumerate(postings):
                for i in range(len(postings[index])):
                    if postings[index][i] == postings[index+1][i]:
                        final_list.append(list_of_docs[i])
                break


            #print(list_of_docs)
            #print(postings)


            return final_list

def boolean_operation(phrase_one, operator, phrase_two):
    matched_phrase1 = phrase_query(phrase_one)
    matched_phrase2 = phrase_query(phrase_two)
    print(matched_phrase1, operator, matched_phrase2)
    
    terms = []
    result = matched_phrase1
    
    if bool(phrase_two) == 0:
        if bool(result) != 0:
            for term in phrase_one:
                terms.append(term)

            for term in phrase_two:
                terms.append(term)
            return terms, result
        else:
            return 0
        
    if operator == "and":
        if bool(matched_phrase2) == 0:
            print("No matched docs for current phrase query")
            return 0
        else:
            result = list(set(matched_phrase1).intersection(set(matched_phrase2)))
            if bool(result) == 0:
                print("No matched docs for current phrase query")
                return 0
            else:
                for term in phrase_one:
                    terms.append(term)

                for term in phrase_two:
                    terms.append(term)
                return terms, result


    elif operator == "or":
        if bool(matched_phrase2) == 0:
            for term in phrase_one:
                terms.append(term)
            return terms, result
        else:
            for term in phrase_one:
                terms.append(term)

            for term in phrase_two:
                terms.append(term)

            for i in matched_phrase2:
                result.append(i)

            return terms, list(set(result))

    elif operator == "not":
        if bool(matched_phrase2) == 0:
            for term in phrase_one:
                terms.append(term)
            return terms, result
        else:
            for i in matched_phrase2:
                if i in result:
                    result.remove(i)

            if bool(result) == 0:
                print("No matched docs for current phrase query")
                return 0
            else:
                for term in phrase_one:
                    terms.append(term)
                return terms, result

# checking similarity scores and ranking
def sim_score(query_stat, doc):
    doc = f"Doc{doc}"
    if doc in query_stat.keys():
        return query_stat.loc["sum", doc]
    else:
        print(f"There is no similarity score between the query and {doc}")
        
def doc_rank(query_stat):
    rank = pd.Series(query_stat.loc["sum"].dropna().sort_values(ascending=False))
    return list(rank.index)


#________________________display part________________________________
print("\n\n")
print("positional_index: \n")
for i in positional_index:
    print(i, positional_index[i])
print("\n\n")

print("tf table: ")
print(tabulate(tf_table, headers='keys', tablefmt='fancy_grid'))
print("\n\n")

print("w_tf table: ")
print(tabulate(w_tf, headers='keys', tablefmt='fancy_grid'))
print("\n\n")

print("df_idf table: ")
print(tabulate(df_idf_table, headers='keys', tablefmt='fancy_grid'))
print("\n\n")

print("tf_idf_table: ")
print(tabulate(tf_idf_table, headers='keys', tablefmt='fancy_grid'))
print("\n\n")

print("doc_length: ")
print(tabulate(doc_length, headers='keys', tablefmt='fancy_grid'))
print("\n\n")

print("normalized_tf_idf: ")
print(tabulate(normalized_tf_idf, headers='keys', tablefmt='fancy_grid'))
print("\n\n")

#_____________________phrase query______________________________

query = input("Please enter phrase query: ")
phrase_one, operator, phrase_two = process_query(query)

if bool(boolean_operation(phrase_one, operator, phrase_two)) != 0:
    query_terms, query_result = boolean_operation(phrase_one, operator, phrase_two)
    query_tf = {}
    if bool(query_result) == 0:
        print("Not results found for query")
    else:
        for term in query_terms:
            if term in query_tf:
                query_tf[term] += 1
            else:
                query_tf[term] = 1
        #print(query_tf)
        query_stat = pd.DataFrame(columns=["tf-raw", "tf(1 + log tf)", "idf", "tf*idf", "Normalized"], index=query_tf.keys())

        query_stat['tf-raw'] = query_tf.values()
        #print(query_stat)

        query_stat["tf(1 + log tf)"] = 1 + query_stat["tf-raw"].apply(log10)

        query_idf = []
        for term in query_terms:
            query_idf.append(float(df_idf_table.loc[term, "idf"]))

        query_stat["idf"] = query_idf
        
        query_stat["tf*idf"] = query_stat["tf(1 + log tf)"] *  query_stat["idf"]
        
        query_length = math.sqrt((query_stat["tf*idf"] **2).sum())
        
        #print(query_length)
        
        query_stat["Normalized"] = query_stat["tf*idf"] / query_length
        
        for doc in query_result:
            query_stat[f"Doc{doc}"] = 0
        
        query_stat.loc["sum"] = np.nan
        
        for query_term in query_terms:
            for i in range(len(query_result)):
                #print(query_term, query_result[i])
                query_stat.loc[query_term, f"Doc{query_result[i]}"] = query_stat.loc[query_term, "Normalized"] * normalized_tf_idf.loc[query_term, f"Doc{query_result[i]}"]
                    
        for doc in query_result:
            query_stat.loc["sum", f"Doc{doc}"] = query_stat[f"Doc{doc}"].sum()


    # print(query_result)
    print("\n\n")
    print("query_stat: ")
    print(tabulate(query_stat, headers='keys', tablefmt='fancy_grid'))
    print(f"Query length: {query_length}")
    print("\n\n")

    # Similarity scores
    checkout = 1

    while checkout:
        choice = input('Enter "s" to check similary score with document number or Enter "r" for the rank: ')

        if choice == "s":
            doc_to_check = input("Enter document number: ")
            print("\n")
            score = sim_score(query_stat, doc_to_check)
            print(f"Similarity score for document {doc_to_check} is: {score}")
            print("\n")

        elif choice == "r":
            print("\n\n")
            d_rank = doc_rank(query_stat)
            print(f"Document rank is: {d_rank}")
            print("\n")
else:
    print("No results found for query")


