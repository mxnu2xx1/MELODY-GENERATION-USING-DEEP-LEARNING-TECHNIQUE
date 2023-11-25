
import os
import music21 as m21
from music21 import *
import json
import numpy as np
import tensorflow.keras as keras

#m21.configure.run()
us = environment.UserSettings()
us['musicxmlPath'] = 'E:\\bin\MuseScore4.exe'
SAVE_DIR="Dataset"
SINGLE_FILE_DATASET="file_dataset"
SEQUENCE_LENGTH=64
MAPPING_PATH="mapping.json"
SAVE_MODEL_PATH="model.h5"
ACCEPTABLE_DURATIONS=[
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4


]
EPOCHS=25
NUM_UNITS=[256]
OUTPUT_UNITS=38
LOSS="SparseCategoricalCrossentropy"
LEARNING_RATE=0.00102
BATCH_SIZE=64
import os
def load_songs_in_kern(dataset_path):
  #print("hi")
  songs=[]
  for path,subdir,files in os.walk(dataset_path):
    print("jhhj")
    for file in files:
      if file[-3:]=="krn":
        song = m21.converter.parse(os.path.join(path,file))
        songs.append(song)
  return songs
def encode_song(song,time_step=0.25):

    encoded_song=[]
    for event in song.flat.notesAndRests:
        if isinstance(event,m21.note.Note):
            symbol=event.pitch.midi
        elif isinstance(event,m21.note.Rest):
            symbol="r"
        steps=int(event.duration.quarterLength/time_step)

        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    encoded_song=" ".join(map(str,encoded_song))
    return encoded_song


def preprocess(dataset_path):
    pass
    print("loading songs....")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    for i,song in enumerate(songs):
        if not has_accpetable_durations(song,ACCEPTABLE_DURATIONS):
            continue
        song=transpose(song)
        encoded_song=encode_song(song)
        save_path = os.path.join(SAVE_DIR,str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)

KERN_DATASET_PATH="deutschl/essen/europa/deutschl/erk"
# Press the green button in the gutter to run the script.

def has_accpetable_durations(song,acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
    parts=song.getElementsByClass(m21.stream.Part)
    measures_part0=parts[0].getElementsByClass(m21.stream.Measure)
    key=measures_part0[0][4]

    if not isinstance(key,m21.key.Key):
        key=song.analyze("key")
    if key.mode=="major":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("C"))
    elif key.mode=="minor":
        interval=m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))
    transposed_song=song.transpose(interval)
    return transposed_song
def load(file_path):
    with open(file_path,"r") as fp:
        song=fp.read()
    return song



def create_single_file_dataset(datasetpath,file_dataset_path,sequence_length):
    new_song_delimiter="/ "*sequence_length
    songs=""
    for path,_,files in os.walk(datasetpath):
        for file in files:
            file_path=os.path.join(path,file)
            song=load(file_path)
            songs=songs+song+" "+new_song_delimiter
    songs=songs[:-1]

    with open(file_dataset_path,"w") as fp:
        fp.write(songs)
    return songs


def create_mappings(songs,mapping_path):
    mappings={}

    songs=songs.split()

    vocabulary=list(set(songs))

    for i,symbols in enumerate(vocabulary):
        mappings[symbols]=i

    with open(mapping_path,"w") as fp:
        json.dump(mappings,fp,indent=4)


def convert_songs_to_int(songs):
    int_songs=[]

    with open(MAPPING_PATH,"r") as fp:
        mappings=json.load(fp)

    songs=songs.split()

    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs
def generate_training_sequences(sequence_length):
    songs=load(SINGLE_FILE_DATASET)
    int_songs=convert_songs_to_int(songs)
    num_sequences=len(int_songs)-sequence_length

    inputs=[]
    targets=[]

    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    vocabulary_size=len(set(int_songs))
    inputs=keras.utils.to_categorical(inputs,num_classes=vocabulary_size)
    targets=np.array(targets)

    return inputs,targets

def build_model(output_units,num_units,loss,learning_rate):
    input=keras.layers.Input(shape=(None,output_units))
    x=keras.layers.LSTM(num_units[0])(input)
    x=keras.layers.Dropout(0.2)(x)
    output=keras.layers.Dense(output_units, activation="softmax")(x)

    model=keras.Model(input,output)

    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=["accuracy"])
    model.summary()

    return model





def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    inputs,targets= generate_training_sequences(SEQUENCE_LENGTH)
    model= build_model(OUTPUT_UNITS,NUM_UNITS, LOSS, LEARNING_RATE)

    model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE)
    model.save(SAVE_MODEL_PATH)



def main():
    #print(KERN_DATASET_PATH)
    #songs = load_songs_in_kern(KERN_DATASET_PATH)
    #print(f"loaded {len(songs)}songs.")
    #song = songs[0]

    #preprocess(KERN_DATASET_PATH)
    #songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    #create_mappings(songs, MAPPING_PATH)

    #inputs,targets=generate_training_sequences(SEQUENCE_LENGTH)


    #print(f"Has acceptable duration? {has_accpetable_durations(song, ACCEPTABLE_DURATIONS)}")
    # song.show()
    #print(f"inputs ={inputs}")
    #print(f"targets = {targets}")
    #transpose(song).show()

    train()



if __name__ == '__main__':
    main()



