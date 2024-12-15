### README: Information Retrieval System

---

#### **Project Overview**
This project implements an **Information Retrieval System** using the **Cosine Similarity** algorithm. The system retrieves and ranks documents based on their relevance to a user-provided query by employing **TF-IDF weighting**.

---

#### **Features**
- **Text Preprocessing**:
  - Tokenization
  - Stop Words Removal
  - Stemming
- **TF (Term Frequency)** Calculation
- **IDF (Inverse Document Frequency)** Calculation
- **TF-IDF Weighting** for each document and query
- **Cosine Similarity** to compute document-query relevance
- **Document Ranking** based on similarity scores

---

#### **Input and Output**
- **Input**:
  - A text file (`LISA.txt`) containing multiple documents (separated by `#`).
  - A user query entered via the console.

- **Output**:
  - Ranked documents based on relevance to the query.
  - Exported Excel files:
    - `docs/tf-file.xlsx`: TF matrix for all documents.
    - `docs/idf-file.xlsx`: IDF values of all words.
    - `docs/tf-idf-file.xlsx`: TF-IDF matrix for all documents.
    - `ranked docs/ranked_documents.xlsx`: Ranked documents with cosine similarity scores.

---

#### **How to Run**
1. Place the `LISA.txt` file in the project directory. This file should contain documents separated by `#`.
2. Run the main script:
   ```bash
   python example.py
   ```
3. Enter your query when prompted.
4. Check the output files in the `docs` and `ranked docs` folders for results.

---

#### **File Structure**
- `LISA.txt`: Input file containing documents.
- `docs/`: Contains TF, IDF, and TF-IDF matrices in Excel format.
- `query/`: Stores TF and TF-IDF matrices for the user query.
- `ranked docs/`: Contains the ranked documents based on cosine similarity.

---

#### **Key Algorithms**
1. **TF-IDF Calculation**:
   - TF-IDF(word) = (1 + log(TF)) × IDF(word)
2. **Cosine Similarity**:
   - Cosine Similarity = Dot Product / (|Query TF-IDF| × |Document TF-IDF|)

---

#### **Example**
- Input Query: `"Information Retrieval"`
- Output: A ranked list of documents sorted by their similarity to the query.

---

#### **Dependencies**
- Python 3.x
- Libraries: 
  - `nltk`
  - `pandas`
  - `openpyxl`
