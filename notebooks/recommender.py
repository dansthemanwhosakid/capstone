import pandas as pd
import numpy as np
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython.display import Image, display, HTML
import time

mov_lines = pd.read_pickle("../data/mov_model.pkl")
mov_info = pd.read_pickle('../data/mov_combo_final.pkl')
model = Doc2Vec.load("../models/d2v.model")

with open('../data/genres.pkl', 'rb') as f:
    unique_genres = pickle.load(f)

def similar_characters(input_char_index):
    print("TOP 10 SIMILAR CHARACTERS")
    # Obtain the top results for character based on index
    similar = np.array(model.docvecs.most_similar(input_char_index, topn = 10))
    similar_characters = list(similar[:,0])
    similar_weights = list(similar[:,1])

    # Reduced dataframe
    df = mov_info.loc[similar_characters, :]
    df.reset_index(drop=True,inplace=True)
    df['similarity'] = similar_weights

    # similar character names:
    similar_characters = list(df['character'].map(lambda x: x.strip()))

    # displaying similar characters and their movie title, text, and scores
    display(df[['imdb_title','character', 'text', 'similarity','genre']])

    print("\nWhich recommended character's lines would you like to examine?")
    print("You can either select the character's id (0 to 9) or input the character name.\n")

    while True:
        choose_line = input("Response: ")
        try:
            if str(choose_line) in similar_characters:
                break
        except:
            print('Error, response not found. Please check your spelling.')
            pass
        try:
            if int(choose_line) in range(10):
                choose_line = int(choose_line)
                break
        except:
            print('Error, response not found. Please check your spelling.')
            pass

    if type(choose_line) == int:

        character_name_to_print = df.loc[int(choose_line),'character']
        print(f'\n{character_name_to_print}')

        text_to_display = df.loc[int(choose_line),'text']
        print(f'"{text_to_display}"')

    elif type(choose_line) == str:
        text_to_display = df[df['character'].str.contains(choose_line)]['text'].values[0]
        print(f'"{text_to_display}"')

    url = mov_info[mov_info['text'] == text_to_display]['imdb_url'].values[0]
    pic = mov_info[mov_info['text'] == text_to_display]['pic_url'].values[0]

    print(f'\nFor more information about the selected movie character, please go to {url}')

    display(Image(url = pic, width = 400, height = 400))

def quick_filter():
    print("Welcome to Dansthemanwhosakid's movie character recommendation extravaganza!!!\n")
    print(f"With over {len(mov_info)} characters, you can select a character of your choice.\n")
    print("You can then see which characters are most similar to your selected character.\n")
    print("Which movie would you like to choose?\n")

    filtered_df = mov_info
    filtered_df['imdb_title'] = filtered_df['imdb_title'].map(lambda x: x.lower())

    time.sleep(2)

    while True:
        choose_mov = str(input("Response: "))
        try:
            if len(filtered_df[filtered_df['imdb_title'].str.contains(choose_mov)]) > 0:
                break
        except:
            print('Error: response not found. Please check your spelling.')
            pass

    mov_mask = (filtered_df['imdb_title'].str.contains(choose_mov))

    filtered_df = filtered_df[mov_mask][['imdb_title','character','text']]

    display(filtered_df)

    character_list = list(filtered_df['character'])

    print("\nWhich character's lines would you like to examine?")
    print("You can either select the character's id or input the character name.\n")

    while True:
        choose_character = input("Response: ")
        try:
            if str(choose_character) in character_list:
                break
        except:
            print('Error, response not found. Please check your spelling.')
            pass
        try:
            if int(choose_character) in list(filtered_df.index):
                choose_character = int(choose_character)
                break
        except:
            print('Error, response not found. Please check your spelling.')
            pass

    if type(choose_character) == int:
        character_id = int(choose_character)
    elif type(choose_character) == str:
        character_id = filtered_df[filtered_df['character'] == choose_character].index[0]

    character_name = mov_info.loc[character_id,'character']
    character_text = mov_info.loc[character_id,'text']

    print(f'\n{character_name}')
    print(f'\n{character_text}')

    url = mov_info[mov_info['text'] == character_text]['imdb_url'].values[0]
    pic = mov_info[mov_info['text'] == character_text]['pic_url'].values[0]

    print(f'\nFor more information about the selected movie character, please go to {url}')

    display(Image(url = pic, width = 400, height = 400))

    similar_characters(character_id)
