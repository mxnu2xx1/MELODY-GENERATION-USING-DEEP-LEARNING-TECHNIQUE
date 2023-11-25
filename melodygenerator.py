import tensorflow.keras as keras
import json
import numpy as np
from main import SEQUENCE_LENGTH,MAPPING_PATH
import music21 as m21
global temp_prob
from nltk.translate.bleu_score import sentence_bleu
class MelodyGenerator:
    def __int__(self,model_path="model.h5"):
        self.model_path="model.h5"
        self.model=keras.models.load_model(model_path)

        with open(MAPPING_PATH,"r") as fp:
            self._mappings=json.load(fp)
        self._start_symbols=["/"]*SEQUENCE_LENGTH




    def generate_melody(self,seed,num_steps,max_sequence_length,temperature):
        seed=seed.split()
        melody=seed
        seed=self._start_symbols +seed

        seed=[self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed=seed[-max_sequence_length:]

            oneHot_seed=keras.utils.to_categorical(seed,num_classes=len(self._mappings))

            oneHot_seed=oneHot_seed[np.newaxis, ...]

            probabilities=self.model.predict(oneHot_seed)[0]
            global temp_prob

            temp_prob=probabilities

            output_int=self.sample_with_temp(probabilities,temperature)

            seed.append(output_int)

            output_symbol=[k  for k,v in self._mappings.items() if v==output_int ][0]

            if output_symbol=="/":
                break
            melody.append(output_symbol)
        return melody

    def sample_with_temp(self,probabilities,temperature):

        global temp_prob
        probabilities=temp_prob



        predictions=np.log(temp_prob)/temperature
        probabiilities=np.exp(predictions)/np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        index=np.random.choice(choices, p=probabilities)


        return index

    def save_melody(self,melody,file_name,format="midi",step_duration=0.25):

        stream=m21.stream.Stream()
        start_symbol=None
        step_counter=1

        for i,symbol in enumerate(melody):

            if symbol !="_" or i+1==len(melody):

                if start_symbol is not None:

                    quarter_length_duration=step_duration * step_counter

                    if start_symbol == "r":
                        m21_event=m21.note.Rest(quarterLength=quarter_length_duration)

                    else:
                        m21_event=m21.note.Note(int(start_symbol),quarterLength=quarter_length_duration)
                    stream.append(m21_event)

                    step_counter=1
                start_symbol=symbol


            else:
                step_counter+=1

        stream.write(format,file_name)

def testing(melody,tester):
    mel=" ".join(melody)
    print(mel)

    print('BLEU score -> {}'.format(sentence_bleu(tester, mel)))




if __name__=="__main__":
    mg=MelodyGenerator()
    seed="55 _ 60 _ 60 _ 62 "
    mg.__int__()
    melody=mg.generate_melody(seed,500,SEQUENCE_LENGTH,0.7)
    #print("hi")
    tester="55 _ 60 _ 60 _ 62 _ 62 65 64 _ 64 62 60 _ 62 64 65 _ 62 _ 60 _ 59 62 57 _ 55 _ r _ 55 _ 60 _ 64 _ 67 _ 60 _ 59 _ 62 _ 65 _ 67 _ 64 _ 64 _ 65 _ 59 _ 60 _ _ _ r _"
    testing(melody,tester)
    print(melody)
    file_name1="mel.mid"
    file_name2="mel1.mid"


    mg.save_melody(melody,file_name1)
    tester=tester.split(" ")
    print(tester)
    mg.save_melody(tester, file_name2)

