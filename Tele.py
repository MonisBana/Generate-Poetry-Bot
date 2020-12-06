import telebot
import pickle
from generatePoetry import predict
from tensorflow.keras.models import load_model

bot = telebot.TeleBot("1400338571:AAH1AhMuZr6ehIdu6sx5iqLebYzqwEs5lwk")
#Load Models
modern_nature = load_model('modern_nature.h5')
renaissance_love = load_model('renaissance_love.h5')
modern_love = load_model('modern_love.h5')
renaissance_nature = load_model('renaissance_nature.h5')

def load_variable(name):
    with open(name+'.pickle', 'rb') as handle:
        variable = pickle.load(handle)
    return variable

#Load Tokenizers
model_nature_tokenizer = load_variable('modern_nature_tokenizer')
renaissance_love_tokenizer = load_variable('renaissance_love_tokenizer')
modern_love_tokenizer = load_variable('modern_love_tokenizer')
renaissance_nature_tokenizer = load_variable('renaissance_nature_tokenizer')

#Load max sequence len
model_nature_max_sequence_len = load_variable('modern_nature_max_sequence_len')
renaissance_love_max_sequence_len = load_variable('renaissance_love_max_sequence_len')
modern_love_max_sequence_len = load_variable('modern_love_max_sequence_len')
renaissance_nature_max_sequence_len = load_variable('renaissance_nature_max_sequence_len')

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")

@bot.message_handler(commands=['modern_nature'])
def genrate_poetry(message):
    input_text = ' '.join(message.text.split(" ")[1:])
    result = predict(modern_nature,modern_love_tokenizer,modern_love_max_sequence_len,input_text,32-len(input_text.split(" ")))
    bot.send_message(message.chat.id ,result)

@bot.message_handler(commands=['renaissance_love'])
def genrate_poetry(message):
    input_text = ' '.join(message.text.split(" ")[1:])
    result = predict(renaissance_love,renaissance_love_tokenizer,renaissance_love_max_sequence_len,input_text,32-len(input_text.split(" ")))
    bot.send_message(message.chat.id ,result)

@bot.message_handler(commands=['modern_love'])
def genrate_poetry(message):
    input_text = ' '.join(message.text.split(" ")[1:])
    result = predict(modern_love,model_nature_tokenizer,modern_love_max_sequence_len,input_text,32-len(input_text.split(" ")))
    bot.send_message(message.chat.id ,result)

@bot.message_handler(commands=['renaissance_nature'])
def genrate_poetry(message):
    input_text = ' '.join(message.text.split(" ")[1:])
    result = predict(renaissance_nature,renaissance_love_tokenizer,renaissance_love_max_sequence_len,input_text,32-len(input_text.split(" ")))
    bot.send_message(message.chat.id ,result)



bot.polling()
