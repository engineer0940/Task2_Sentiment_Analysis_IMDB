# Task2_Sentiment_Analysis_IMDB
The task implemented is a sentiment analysis using Recurrent Neural Networks (RNNs) on the IMDB movie reviews dataset. The goal is to classify movie reviews as positive or negative based on their content. Sentiment analysis is a common natural language processing (NLP) task used to identify and categorize opinions expressed in a piece of text.

# Dataset
The IMDB dataset consists of 50,000 movie reviews, with 25,000 for positive and 25,000 for negative reviews. The dataset is widely used for binary sentiment classification tasks.

# Preprocessing
Preprocessing steps are crucial to prepare the textual data for the RNN model. The following preprocessing steps were performed:
Cleaning of text: html tags removal, conversion to lower case, removal of stop words and punctuation and lemmatization steps are applied to clean the reviews text.
Tokenization: Converting the text into sequences of integers where each integer represents a specific word in the vocabulary.
Padding: Ensuring all sequences have the same length by padding shorter sequences with zeros and truncating longer sequences. This is necessary for efficient batch processing in RNNs.
Vocabulary Size: The vocabulary size was set to 5,000, which means only the top 5,000 most frequent words in the dataset were considered. This helps in reducing the complexity and memory usage.
Maxlen of Reviews: The maximum length of the reviews was set to 200 words. Reviews longer than this were truncated, and shorter reviews were padded with zeros.
Selection of Parameters
Vocabulary Size: Set to 5,000 to include the most frequent words and reduce the size of the input space.
Maxlen: Set to 200 to maintain a balance between including enough context and keeping the computation feasible.
# Training, validation and Test datasets
Training examples: 39200
Validation: 800
Test examples: 10000
# Different Learning Models Used
Several RNN architectures were experimented with to compare their performance:

Simple RNN: A basic RNN architecture without embeddings with a single RNN layer followed by dense layers.
Simple RNN with embeddings: A basic RNN architecture with embeddings layer of size 128 as input to a single RNN layer followed by dense layers.
LSTM (Long Short-Term Memory): An embedding layer followed by advanced RNN variant designed to handle long-term dependencies better than simple RNNs.
GRU (Gated Recurrent Unit): Another advanced RNN variant similar to LSTM but with a slightly different architecture that is computationally more efficient.

# Comparison of Their Accuracy
The models were trained and evaluated on the IMDB dataset.embedding size is 128, batch size is 64 and number of epochs is 5. The accuracy of each model on the test set was compared to determine the best-performing architecture.

Model	Accuracy
Simple RNN	49.75%
Simple RNN with embedding layer 85.07%
GRU	85.74%
LSTM	86.79%

The LSTM model achieved the highest accuracy, demonstrating its effectiveness in capturing long-term dependencies in the text.

# Testing the Model on New Reviews
To further validate the model, it was tested on new, unseen reviews. The model successfully classified the sentiment of the new reviews, indicating its robustness and generalization capability.

# Conclusion
In this sentiment analysis task using RNN on the IMDB dataset:

The LSTM model outperformed the Simple RNN and GRU models with an accuracy of 86.79%.
Preprocessing steps like tokenization and padding were crucial for handling the textual data.
The chosen parameters, such as vocabulary size and maxlen, balanced computational efficiency and model performance.
The model was effective in classifying the sentiment of both the test set and new reviews.
This implementation demonstrates the power of RNNs, especially LSTMs, in NLP tasks such as sentiment analysis.
