import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import math
import pandas as pd

def read_text_files(folder_path):
    texts = {}
    try:
        with open(folder_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text_parts = text.split("#")
            for i, part in enumerate(text_parts):
                texts[folder_path + f"_{i+1}"] = part.strip()  # Store the texts in a dictionary
        # print(texts)        
        return texts
    except Exception as e:
        print("Error reading text files:", str(e))
        return None

folder_path = "LISA.txt"
texts = read_text_files(folder_path)

nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()

def preprocess_text(texts):
    try:
        stop_words = set(stopwords.words('english'))  # List of stop words
        tokenized_texts = {}
        for key, text in texts.items():
            # print(key)
            words = word_tokenize(text)  # Tokenization
            filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words and word.isalnum()]  # Remove stop words and perform stemming
            tokenized_texts[key] = filtered_words
        return tokenized_texts
    except Exception as e:
        print("Error preprocessing text:", str(e))
        return None

tokenized_texts = preprocess_text(texts)

def count_word_occurrences(tokenized_texts):
    try:
        word_occurrences = {}
        for key, words in tokenized_texts.items():  # Find the word count for each document
            occurrences = {}
            for word in words:
                if word in occurrences:
                    occurrences[word] += 1
                else:
                    occurrences[word] = 1
            word_occurrences[key] = occurrences
        return word_occurrences
    except Exception as e:
        print("Error counting word occurrences:", str(e))
        return None

word_occurrences = count_word_occurrences(tokenized_texts)

def create_word_document_matrix(word_occurrences):
    try:
        # Find all words in all documents
        all_words = set()
        for occurrences in word_occurrences.values():
            # print(occurrences)
            all_words.update(occurrences.keys())

        # Create the matrix
        matrix = []
        # print(word_occurrences.items())
        for doc, occurrences in word_occurrences.items():
            row = []
            for word in all_words:
                if word in occurrences:
                    row.append(1 + math.log10(occurrences[word])) #tf calculate
                else:
                    row.append(0)
            matrix.append(row)
        
        # Add document names to the matrix
        document_names = list(word_occurrences.keys())
        for i, row in enumerate(matrix):
            matrix[i] = [document_names[i]] + row

        # Add word names to the columns of the matrix
        all_words = ['Document'] + list(all_words)
        matrix.insert(0, all_words)

        return matrix
    except Exception as e:
        print("Error creating word-document matrix:", str(e))
        return None

word_document_matrix = create_word_document_matrix(word_occurrences)

# Convert word-document matrix to DataFrame
def matrix_to_dataframe(matrix):
    try:
        df = pd.DataFrame(matrix[1:], columns=matrix[0])  # The first row as column names
        return df
    except Exception as e:
        print("Error converting matrix to DataFrame:", str(e))
        return None

word_document_df = matrix_to_dataframe(word_document_matrix)
word_document_df.to_excel("docs/tf-file.xlsx")

def calculate_idf(word_occurrences, num_documents):
    idf_values = {}
    for occurrences in word_occurrences.values():
        for word in occurrences.keys():
            if word in idf_values:
                idf_values[word] += 1
            else:
                idf_values[word] = 1

    for word, count in idf_values.items():
        idf_values[word] = math.log10(num_documents / count)

    return idf_values

# Number of documents
num_documents = len(word_occurrences)

# Calculate IDF
idf_values = calculate_idf(word_occurrences, num_documents)

# Create word-IDF matrix
def create_word_idf_matrix(idf_values):
    try:
        # Find all words in all documents
        all_words = list(idf_values.keys())

        # Create the matrix
        matrix = []
        matrix.append(all_words)  # First row: all words
        matrix.append([idf_values[word] for word in all_words])  # Second row: IDF value of each word

        return matrix
    except Exception as e:
        print("Error creating word-IDF matrix:", str(e))
        return None

word_idf_matrix = create_word_idf_matrix(idf_values)

# Convert word-IDF matrix to DataFrame
word_idf_df = matrix_to_dataframe(word_idf_matrix)
word_idf_df.to_excel("docs/idf-file.xlsx")

def calculate_tf_idf(word_occurrences, idf_values):
    tf_idf_matrix = {}

    for doc, occurrences in word_occurrences.items():
        tf_idf_values = {}
        for word, tf in occurrences.items():
            tf_idf_values[word] = (1 + math.log10(tf)) * idf_values[word]
        tf_idf_matrix[doc] = tf_idf_values

    return tf_idf_matrix

tf_idf_matrix = calculate_tf_idf(word_occurrences, idf_values)

def create_tf_idf_dataframe(tf_idf_matrix):
    try:
        # Find all words in all documents
        all_words = set()
        for tf_idf_values in tf_idf_matrix.values():
            all_words.update(tf_idf_values.keys())

        # Create the matrix
        matrix = []
        for doc, tf_idf_values in tf_idf_matrix.items():
            row = []
            for word in all_words:
                if word in tf_idf_values:
                    row.append(tf_idf_values[word])
                else:
                    row.append(0)
            matrix.append(row)

        # Add document names to the matrix
        document_names = list(tf_idf_matrix.keys())
        for i, row in enumerate(matrix):
            matrix[i] = [document_names[i]] + row

        # Add word names to the columns of the matrix
        all_words = ['Document'] + list(all_words)
        matrix.insert(0, all_words)

        df = pd.DataFrame(matrix[1:], columns=matrix[0])
        return df
    except Exception as e:
        print("Error creating TF-IDF DataFrame:", str(e))
        return None

tf_idf_df = create_tf_idf_dataframe(tf_idf_matrix)
tf_idf_df.to_excel("docs/tf-idf-file.xlsx")

# Enter query from user
input_query = input("please enter your text: \n")
query = {"query": input_query}
# Tokenize query
tokenized_query = preprocess_text(query)
# TF of query
tf_query = count_word_occurrences(tokenized_query)
# Matrix TF of query
tf_matrix_query = create_word_document_matrix(tf_query)
# Matrix TF of query to DataFrame
tf_dataframe_query = matrix_to_dataframe(tf_matrix_query)
# TF to Excel
tf_dataframe_query.to_excel("query/tf-query-file.xlsx")

# Calculate IDF
def calculate_query_idf(tf_query, word_occurrences, num_documents):
    query_idf_values = {}
    # print(tf_query['query'])
    # Count the number of documents that contain each word in the query
    for word in tf_query['query'].keys():
        count = sum(1 for occurrences in word_occurrences.values() if word in occurrences)
        
        # If a word does not appear in any document, set its count to 1
        if count == 0:
            count = 1
        
        # Calculate IDF
        query_idf_values[word] = math.log10(num_documents / count)
    
    return query_idf_values

idf_query_values = calculate_query_idf(tf_query, word_occurrences, num_documents)

# Convert IDF to matrix
idf_query_matrix = create_word_idf_matrix(idf_query_values)

# Convert matrix to DataFrame
idf_query_dataframe = matrix_to_dataframe(idf_query_matrix)
idf_query_dataframe.to_excel("query/idf-query-file.xlsx")

# Calculate TF-IDF of query
tf_idf_query = calculate_tf_idf(tf_query, idf_query_values)

# Convert TF-IDF matrix to DataFrame
tf_idf_query_dataframe = create_tf_idf_dataframe(tf_idf_query)
tf_idf_query_dataframe.to_excel("query/tf-idf-query-file.xlsx")

# calculate cosine similarity of query tf-idf and docs tf-idf
def calculate_cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[word] * vec2[word] for word in vec1 if word in vec2)
    
    magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

# Calculate cosine similarity
cosine_similarities = {}
for doc, tf_idf_doc in tf_idf_matrix.items():
    
    cosine_similarities[doc] = calculate_cosine_similarity(tf_idf_query["query"], tf_idf_doc)

# Rank documents by similarity
ranked_docs = sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True)

# Output ranked documents
for doc, similarity in ranked_docs:
    print(f"Document: {doc}, Cosine Similarity: {similarity}")

# Create a matrix for ranked documents
ranked_matrix = [["Document", "Cosine Similarity"]]
for doc, similarity in ranked_docs:
    ranked_matrix.append([doc, similarity])

# Convert the matrix to a DataFrame
ranked_df = pd.DataFrame(ranked_matrix[1:], columns=ranked_matrix[0])

# Save the DataFrame to an Excel file
ranked_df.to_excel("ranked docs/ranked_documents.xlsx", index=False)