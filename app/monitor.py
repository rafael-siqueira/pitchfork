from review_utils import *

def monitor():
    new_reviews_df = get_new_reviews(size)
    save_reviews(new_reviews_df, review_model, word_to_index, max_review_length)
    return True
    
# load model and GloVe embeddings
review_model = load_model('./best_model_3.hdf5')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./glove.6B.50d.txt')
del index_to_word
del word_to_vec_map
max_review_length = 1500
size = 6

if __name__ == "__main__":
    monitor()
