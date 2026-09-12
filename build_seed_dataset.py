"""
EduGuide - Seed QA Dataset Builder
===================================
Produces the hand-curated seed set of the education_dataset.csv described in the
project plan. This seed set is written by hand (not scraped/LLM-generated) so that
the project has a real, quality-controlled foundation before scaling up with the
scraping + templated-generation pipeline in collect_docs.py (Phase 2, step 2).

Why a hand-curated seed first:
- Guarantees at least one clean, correct, well-formed example per topic/difficulty
  cell before any automated collection runs (a documented data-quality baseline).
- Gives you something to test the cleaning/EDA/chunking pipeline against
  immediately, without needing internet access.
- Doubles as a small held-out "gold" reference set you can spot-check automated
  QA-pair generation against later, since it is highest-confidence in this dataset.

Run this locally or in Colab -> writes data/raw/education_dataset_seed.csv
"""

import pandas as pd
import os

# Each row: (topic, subtopic, question, context, answer, difficulty, source, source_url)
# context = the short passage the answer should be grounded in (used for RAG retrieval too)
ROWS = [
    # ---------------- Python ----------------
    ("Python", "Data Types", "What is a Python list?",
     "Lists are one of Python's built-in data structures used to store an ordered "
     "collection of items. Lists are mutable, meaning items can be added, removed, "
     "or changed after creation.",
     "A list is an ordered, mutable collection of items in Python, created with square "
     "brackets, e.g. my_list = [1, 2, 3]. You can change, add, or remove elements after "
     "creation.",
     "Beginner", "Python Documentation", "https://docs.python.org/3/tutorial/datastructures.html"),

    ("Python", "Data Types", "What is the difference between a list and a tuple?",
     "Tuples are similar to lists but are immutable: once created, their contents "
     "cannot be changed. Lists are mutable and support methods like append() and "
     "remove() that tuples do not.",
     "A list is mutable (its elements can be changed after creation), while a tuple is "
     "immutable (its elements are fixed once created). Tuples are typically used for "
     "fixed collections of items, like coordinates, while lists are used when the "
     "collection may need to grow or change.",
     "Beginner", "Python Documentation", "https://docs.python.org/3/tutorial/datastructures.html"),

    ("Python", "Comprehensions", "What is list comprehension?",
     "List comprehensions provide a concise way to create lists. Common applications "
     "are to make new lists where each element is the result of some operation applied "
     "to each member of another sequence.",
     "List comprehension is a compact syntax for building a new list from an existing "
     "iterable in a single line, e.g. squares = [x**2 for x in range(10)] instead of a "
     "for-loop with .append(). It can also include a condition, e.g. "
     "[x for x in range(10) if x % 2 == 0].",
     "Intermediate", "Python Documentation", "https://docs.python.org/3/tutorial/datastructures.html"),

    ("Python", "Functions", "What is the difference between *args and **kwargs?",
     "*args allows a function to accept any number of positional arguments, collected "
     "into a tuple. **kwargs allows a function to accept any number of keyword "
     "arguments, collected into a dictionary.",
     "*args lets a function accept a variable number of positional arguments (collected "
     "as a tuple), while **kwargs lets it accept a variable number of keyword arguments "
     "(collected as a dictionary). Example: def f(*args, **kwargs): ... allows calls "
     "like f(1, 2, name='x').",
     "Intermediate", "Python Documentation", "https://docs.python.org/3/tutorial/controlflow.html"),

    ("Python", "Exceptions", "How does exception handling work in Python?",
     "The try statement allows you to test a block of code for errors, the except "
     "block lets you handle the error, and the finally block lets you execute code "
     "regardless of the result.",
     "Python uses try/except blocks to handle errors: code that might raise an error "
     "goes in the try block, and the except block catches and handles specific "
     "exception types, e.g. try: 1/0 except ZeroDivisionError: print('cannot divide "
     "by zero').",
     "Beginner", "Python Documentation", "https://docs.python.org/3/tutorial/errors.html"),

    # ---------------- NumPy / pandas ----------------
    ("Python", "NumPy", "What is a NumPy array and how is it different from a list?",
     "NumPy's main object is the homogeneous multidimensional array, ndarray. Unlike "
     "Python lists, NumPy arrays require all elements to be the same type, which "
     "enables fast, vectorized operations.",
     "A NumPy array (ndarray) is a fixed-type, multi-dimensional grid of values, unlike "
     "a Python list which can hold mixed types. Arrays support fast vectorized math "
     "(e.g. arr * 2 multiplies every element) without writing explicit loops, which "
     "makes NumPy much faster for numerical computation.",
     "Beginner", "NumPy Documentation", "https://numpy.org/doc/stable/user/absolute_beginners.html"),

    ("Python", "pandas", "What is the difference between a pandas Series and a DataFrame?",
     "A Series is a one-dimensional labeled array. A DataFrame is a two-dimensional "
     "labeled data structure with columns of potentially different types, essentially "
     "a collection of Series sharing the same index.",
     "A pandas Series is a single labeled column of data (1-dimensional), while a "
     "DataFrame is a table made of multiple Series sharing the same row index "
     "(2-dimensional), similar to a spreadsheet or SQL table.",
     "Beginner", "pandas Documentation", "https://pandas.pydata.org/docs/getting_started/intro_tutorials/"),

    ("Python", "pandas", "Show me Python code for reading a CSV file using pandas.",
     "pandas.read_csv() reads a comma-separated values file into a DataFrame.",
     "Use pandas.read_csv():\n\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\n\n"
     "This loads the file into a DataFrame called df and head() previews the first 5 rows.",
     "Beginner", "pandas Documentation", "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html"),

    ("Python", "pandas", "How do I handle missing values in a pandas DataFrame?",
     "pandas represents missing data as NaN. Common approaches are dropna() to remove "
     "rows/columns with missing values, or fillna() to replace them with a specific "
     "value or a computed statistic like the mean.",
     "You can either drop missing values with df.dropna(), or fill them in with "
     "df.fillna(value) — for example df['col'].fillna(df['col'].mean()) to fill with "
     "the column mean. Which to use depends on how much data is missing and whether "
     "dropping rows would lose too much information.",
     "Intermediate", "pandas Documentation", "https://pandas.pydata.org/docs/user_guide/missing_data.html"),

    # ---------------- Machine Learning ----------------
    ("Machine Learning", "Model Evaluation", "What is overfitting in machine learning?",
     "Overfitting occurs when a model learns the training data, including its noise "
     "and random fluctuations, too closely, and as a result performs poorly on new, "
     "unseen data.",
     "Overfitting happens when a model fits the training data too closely, including "
     "noise, so it performs well on training data but poorly on new data. A classic "
     "sign is high training accuracy but much lower validation/test accuracy. Common "
     "prevention techniques include cross-validation, regularization, simplifying the "
     "model, and gathering more training data.",
     "Beginner", "scikit-learn Documentation", "https://scikit-learn.org/stable/modules/learning_curve.html"),

    ("Machine Learning", "Model Evaluation", "What is cross-validation?",
     "Cross-validation is a resampling method that uses different portions of the data "
     "to test and train a model on different iterations, typically to get a more "
     "reliable estimate of model performance than a single train/test split.",
     "Cross-validation splits the data into multiple folds, trains the model on some "
     "folds and evaluates it on the remaining fold, then repeats this so every fold is "
     "used for evaluation once. Averaging the results gives a more reliable performance "
     "estimate than a single train/test split, and helps detect overfitting.",
     "Intermediate", "scikit-learn Documentation", "https://scikit-learn.org/stable/modules/cross_validation.html"),

    ("Machine Learning", "Model Evaluation", "What is the difference between precision and recall?",
     "Precision is the ratio of true positives to all predicted positives. Recall is "
     "the ratio of true positives to all actual positives. There is typically a "
     "trade-off between the two.",
     "Precision answers 'of everything I predicted positive, how many were actually "
     "positive?' while recall answers 'of everything that was actually positive, how "
     "many did I correctly catch?'. High precision means few false alarms; high recall "
     "means few missed cases. A model can have high accuracy but still perform poorly "
     "if it does well on the majority class while missing most of a rare but important "
     "minority class — which is why precision/recall matter especially on imbalanced data.",
     "Intermediate", "scikit-learn Documentation", "https://scikit-learn.org/stable/modules/model_evaluation.html"),

    ("Machine Learning", "Learning Types", "What is the difference between supervised and unsupervised learning?",
     "Supervised learning uses labeled training data to learn a mapping from inputs to "
     "known outputs. Unsupervised learning works with unlabeled data to find patterns "
     "or structure, such as clusters, without predefined labels.",
     "In supervised learning, the model learns from labeled examples (input-output "
     "pairs) to predict outputs for new inputs, e.g. classifying emails as spam or not. "
     "In unsupervised learning, there are no labels — the model finds structure on its "
     "own, e.g. clustering customers into segments based on purchasing behavior.",
     "Beginner", "scikit-learn Documentation", "https://scikit-learn.org/stable/unsupervised_learning.html"),

    ("Machine Learning", "Algorithms", "What is a decision tree?",
     "A decision tree is a model that splits data into branches based on feature "
     "values, forming a tree structure, to arrive at a prediction at each leaf.",
     "A decision tree predicts an outcome by repeatedly splitting the data based on "
     "feature values (e.g. 'is age > 30?'), forming a tree of yes/no questions that "
     "leads to a final prediction at the leaf nodes. They are easy to interpret but can "
     "overfit if grown too deep, which is why techniques like pruning or ensembles "
     "(e.g. random forests) are often used.",
     "Beginner", "scikit-learn Documentation", "https://scikit-learn.org/stable/modules/tree.html"),

    ("Machine Learning", "Model Selection", "Why can a model have high accuracy but still perform poorly?",
     "Accuracy can be misleading on imbalanced datasets, where predicting the majority "
     "class most of the time yields high accuracy despite poor performance on the "
     "minority class.",
     "On an imbalanced dataset (e.g. 95% of examples belong to one class), a model that "
     "always predicts the majority class scores 95% accuracy while being useless at "
     "identifying the minority class. This is why metrics like precision, recall, F1 "
     "score, or a confusion matrix are used alongside accuracy, especially when the "
     "minority class matters most (e.g. detecting fraud or disease).",
     "Intermediate", "scikit-learn Documentation", "https://scikit-learn.org/stable/modules/model_evaluation.html"),

    # ---------------- NLP ----------------
    ("NLP", "Text Preprocessing", "What is tokenization?",
     "Tokenization is the process of splitting text into smaller units, called tokens, "
     "such as words or subwords, as a preprocessing step for NLP models.",
     "Tokenization splits raw text into smaller units (tokens) that a model can process "
     "— typically words, subwords, or characters. For example, 'I love NLP' might "
     "tokenize into ['I', 'love', 'NLP']. Modern LLMs typically use subword "
     "tokenization (e.g. Byte-Pair Encoding) so rare words are split into familiar "
     "pieces rather than being treated as unknown.",
     "Beginner", "Hugging Face Documentation", "https://huggingface.co/docs/transformers/tokenizer_summary"),

    ("NLP", "Text Representation", "What is TF-IDF?",
     "TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that "
     "reflects how important a word is to a document within a collection of documents, "
     "weighting terms that are frequent in a document but rare across the corpus.",
     "TF-IDF scores a word by how often it appears in a document (term frequency) "
     "weighted down by how common it is across all documents (inverse document "
     "frequency) — so common words like 'the' get low scores while distinctive words "
     "specific to a document get high scores. It's often used as a simple baseline for "
     "text search and retrieval before moving to embedding-based methods.",
     "Intermediate", "scikit-learn Documentation", "https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting"),

    ("NLP", "Embeddings", "What are word embeddings?",
     "Word embeddings are dense vector representations of words that capture semantic "
     "meaning, such that words with similar meanings have similar vectors.",
     "Word embeddings represent words (or sentences, in the case of sentence "
     "embeddings) as dense numeric vectors positioned so that semantically similar "
     "items are close together in the vector space. This is what powers similarity "
     "search in a RAG system: a user's question is embedded and compared against "
     "embedded document chunks to find the most relevant context.",
     "Intermediate", "Sentence-Transformers Documentation", "https://www.sbert.net/"),

    # ---------------- Deep Learning ----------------
    ("Deep Learning", "Training", "What is backpropagation?",
     "Backpropagation is the algorithm used to train neural networks by computing the "
     "gradient of the loss function with respect to each weight, propagating the error "
     "backward from the output layer to the input layer.",
     "Backpropagation computes how much each weight in a neural network contributed to "
     "the prediction error, working backward from the output layer, using the chain "
     "rule of calculus. Those gradients are then used by an optimizer (e.g. gradient "
     "descent) to update the weights and reduce the error on the next pass.",
     "Intermediate", "Deep Learning Educational Resources", "https://www.deeplearningbook.org/"),

    ("Deep Learning", "Architectures", "What is a neural network, explained simply?",
     "A neural network is a computing system loosely inspired by biological brains, "
     "made of layers of connected nodes ('neurons') that transform input data through "
     "weighted connections to produce an output.",
     "At a basic level, a neural network takes input data (e.g. pixel values of an "
     "image), passes it through layers of simple mathematical units called neurons, "
     "each applying a weighted sum and a small nonlinear transformation, and produces "
     "an output (e.g. 'cat' or 'dog'). The network learns by adjusting the weights on "
     "each connection based on how wrong its predictions were during training.",
     "Beginner", "Deep Learning Educational Resources", "https://www.deeplearningbook.org/"),

    ("Deep Learning", "Architectures", "What is a neural network, explained at an intermediate level?",
     "A neural network learns a function by composing layers of affine transformations "
     "(weights and biases) with nonlinear activation functions, with parameters "
     "optimized via backpropagation and gradient-based optimization.",
     "A neural network is a stack of layers, each computing output = activation(W·x + b) "
     "for weight matrix W, bias b, and input x. Nonlinear activation functions (ReLU, "
     "sigmoid, etc.) let the network model complex, non-linear relationships rather than "
     "just a linear function. Training uses backpropagation to compute gradients of a "
     "loss function with respect to every weight, and an optimizer (e.g. Adam or SGD) "
     "updates the weights to minimize that loss over many iterations.",
     "Intermediate", "Deep Learning Educational Resources", "https://www.deeplearningbook.org/"),

    ("Deep Learning", "Regularization", "What is dropout in neural networks?",
     "Dropout is a regularization technique where randomly selected neurons are "
     "ignored (dropped out) during training, which helps prevent overfitting.",
     "Dropout randomly disables a fraction of neurons during each training step, "
     "forcing the network to not rely too heavily on any single neuron. This acts as a "
     "regularizer and reduces overfitting, similar in spirit to training an ensemble "
     "of smaller networks. Dropout is turned off at inference/evaluation time.",
     "Intermediate", "Deep Learning Educational Resources", "https://www.deeplearningbook.org/"),

    # ---------------- Data Science / General Programming ----------------
    ("Data Science", "Workflow", "What are the typical steps in a data science project?",
     "A typical data science workflow includes problem definition, data collection, "
     "data cleaning, exploratory data analysis, feature engineering, model building, "
     "evaluation, and deployment.",
     "A typical project moves through: (1) defining the problem/business question, "
     "(2) collecting relevant data, (3) cleaning and preprocessing it, "
     "(4) exploratory data analysis to understand patterns, (5) feature engineering, "
     "(6) building and training models, (7) evaluating performance against a metric "
     "tied to the business goal, and (8) deploying and monitoring the model.",
     "Beginner", "General Data Science Educational Resources", "https://www.kaggle.com/learn"),

    ("Data Science", "Statistics", "What is a p-value?",
     "A p-value is the probability of observing results at least as extreme as those "
     "measured, assuming the null hypothesis is true. A small p-value suggests the "
     "observed effect is unlikely to be due to chance alone.",
     "A p-value measures how surprising your observed data would be if there were "
     "truly no effect (the null hypothesis). A small p-value (commonly < 0.05) means "
     "the observed result would be unlikely under the null hypothesis, so it's taken "
     "as evidence against it — but a p-value alone doesn't tell you the size or "
     "practical importance of an effect.",
     "Intermediate", "General Data Science Educational Resources", "https://www.khanacademy.org/math/statistics-probability"),

    ("General Programming", "Version Control", "What is Git and why is it used?",
     "Git is a distributed version control system that tracks changes to files over "
     "time, allowing multiple people to collaborate on a codebase and revert to "
     "previous versions if needed.",
     "Git tracks every change made to a project's files over time, letting you save "
     "checkpoints ('commits'), work on separate features without interfering with "
     "each other ('branches'), and merge everyone's work back together. It's the "
     "standard tool for collaborating on code and for safely experimenting, since you "
     "can always roll back to a previous working version.",
     "Beginner", "Git Documentation", "https://git-scm.com/doc"),

    ("General Programming", "Debugging", "What's a good strategy for debugging code that isn't working?",
     "Effective debugging involves reproducing the issue reliably, isolating the "
     "smallest failing case, checking assumptions with print statements or a debugger, "
     "and verifying the fix doesn't break other functionality.",
     "A good approach: (1) reproduce the bug consistently, (2) narrow it down to the "
     "smallest piece of code that still fails, (3) inspect intermediate values with "
     "print statements or a debugger rather than guessing, (4) check your assumptions "
     "about what the code should be doing at each step, and (5) once fixed, re-run "
     "related tests to make sure nothing else broke.",
     "Beginner", "General Programming Educational Resources", "https://docs.python.org/3/library/pdb.html"),
]

COLUMNS = ["id", "topic", "subtopic", "question", "context", "answer", "difficulty", "source", "source_url"]

def build():
    records = []
    for i, row in enumerate(ROWS, start=1):
        topic, subtopic, question, context, answer, difficulty, source, source_url = row
        records.append({
            "id": f"seed_{i:03d}",
            "topic": topic,
            "subtopic": subtopic,
            "question": question,
            "context": context,
            "answer": answer,
            "difficulty": difficulty,
            "source": source,
            "source_url": source_url,
        })
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = build()
    out_path = "data/raw/education_dataset_seed.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} seed QA pairs to {out_path}")
    print("\nCoverage by topic:")
    print(df["topic"].value_counts())
    print("\nCoverage by difficulty:")
    print(df["difficulty"].value_counts())
