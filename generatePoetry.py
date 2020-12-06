from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

models = ['modern_nature','renaissance_love','modern_love','renaissance_nature']


def save_variable(name,variable):
    with open(name+'.pickle', 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)



def predict(model,tokenizer,max_sequence_len,seed_text,next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word 
    
    temp_seed_text = seed_text.split(" ")
    pred = ''
    for i in range(0,len(temp_seed_text),8):
        pred += ' '.join(temp_seed_text[i:i+8]) + '\n'

    return pred 

