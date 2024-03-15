from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pytube import YouTube
from pydub import AudioSegment
import youtube_dl
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import xgboost
import joblib
import random
import re
import spacy
from spacy.matcher import Matcher
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from firebase import firebase
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from pytube import YouTube
from googleapiclient.discovery import build
from flask import Flask, request, jsonify
from flask_cors import CORS 

nlp = spacy.load("en_core_web_sm")
# spacy.cli.download("en_core_web_sm")

load_dotenv()

DEBUG_LEVEL = os.getenv('DEBUG_LEVEL')
print("DEBUG_LEVEL: "+str(DEBUG_LEVEL))


#model
df = pd.read_csv(r'music_data/features_3_sec.csv')
df = df[['chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','tempo','label']]
label_encoder = LabelEncoder()
df['label'] =  label_encoder.fit_transform(df['label'])

target_name = ['blues', 'classical', 'country', 'disco', 'hiphop' ,'jazz' ,'metal', 'pop','reggae' ,'rock']

xgb = joblib.load('model.pkl')
cols_when_model_builds = xgb.get_booster().feature_names




#find sim music function
def find_sim(data):
    placeHoldername = 'test'
    data['filename'] = placeHoldername

    df_sim = pd.read_csv(r'music_data/features_30_sec.csv')

    df_sim = df_sim[['filename','chroma_stft_mean','chroma_stft_var','rms_mean','rms_var','spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean','zero_crossing_rate_var','harmony_mean','harmony_var','tempo','label']]


    label_encoder = LabelEncoder()
    df_sim['label'] = df_sim['label'].astype("string")
    df_sim['label'] =  label_encoder.fit_transform(df_sim['label'])




    combined_df = pd.concat([df_sim, data], ignore_index=True)

    combined_df = combined_df.set_index('filename')

 
    
    similarity = cosine_similarity(combined_df)

    sim_df_names = pd.DataFrame(similarity, columns=combined_df.index, index=combined_df.index)

    series = sim_df_names[placeHoldername].sort_values(ascending=False)
    series = series.drop(placeHoldername)

    series = series.head(3).to_dict()


    k = Counter(series)
    
    # Finding 3 highest values
    series = k.most_common(3) 
    
    for i in series:
        print(i[0]," :",i[1]," ")

    return series


#users linear regression to predict features of previusly liked music
def find_pred(result):


    model = LinearRegression()

    dfs = []

    if(result is None):
        return None
    

    df = pd.DataFrame.from_dict(result, orient='index')
    dfs.append(df)


    combined_df = pd.concat(dfs, ignore_index=True)

    newData=[]
    for i in range(len(df)):
        y = json.loads(combined_df[0][i])
        newData.append(y)

    if(len(newData)<3):
        return None

    newData = pd.DataFrame(newData)

    finalDf = pd.DataFrame()

    for i in range(len(newData)):
        df = pd.DataFrame([newData[0][i]])
        finalDf = pd.concat([finalDf, df], ignore_index=True)

    finalDf = finalDf.drop('filename', axis=1)


    features = pd.DataFrame(columns=finalDf.columns)

    for column in finalDf:
        X_train, X_test, y_train, y_test = train_test_split(finalDf, finalDf[column], test_size=0.2)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        features.loc[0, column] = prediction[0]
        # print(column,":",prediction[0])
        
    return features




#returns the highest scores
def confidence_score(proba):
    confi = {}
    i=0
    for val in proba[0]:
        rounded = round(val *100,2)
        confi[target_name[i]] = rounded
        i = i+1

    counter = Counter(confi)
    
    high = counter.most_common(3) 
    return high


#extract mfeatures from a song
def extract_features(file):
    y, sr = librosa.load(file,  duration=30)

    chroma_sft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmony = librosa.effects.harmonic(y)
    tempo = librosa.feature.tempo(y=y, sr=sr)[0]

    features = pd.DataFrame({'chroma_stft_mean':[chroma_sft.mean()],'chroma_stft_var':[chroma_sft.var()],'rms_mean':[rms.mean()],'rms_var':[rms.var()],'spectral_centroid_mean':[spectral_centroid.mean()],
                             'spectral_centroid_var':[spectral_centroid.var()],'spectral_bandwidth_mean':[spectral_bandwidth.mean()],'spectral_bandwidth_var':[spectral_bandwidth.var()],
                             'rolloff_mean':[rolloff.mean()],'rolloff_var':[rolloff.var()],'zero_crossing_rate_mean':[zero_crossing_rate.mean()],'zero_crossing_rate_var':[zero_crossing_rate.var()],
                             'harmony_mean':[harmony.mean()],'harmony_var':[harmony.var()],'tempo':[tempo],})
    

    features = features.reindex(columns=cols_when_model_builds)

    return features

#search youtube for a song
def search(query):
    api_key = os.getenv('YOUTUBE_API_KEY')

    youtube = build('youtube', 'v3', developerKey=api_key)

    search_response = youtube.search().list(q=query,type='video',part='id,snippet',maxResults=1).execute()



    for search_result in search_response.get('items', []):
        video_id = search_result['id']['videoId']
        # video_title = search_result['snippet']['title']
        video_url = f'https://www.youtube.com/watch?v={video_id}'

        #print(f'Video Ulr: {video_url}')
        url = video_url

    video = YouTube(url)

    video_duration_seconds = video.length
    video_duration_minutes = video_duration_seconds / 60

    if video_duration_minutes < 6:
        stream = video.streams.filter(only_audio=True).first()
        stream.download(filename=f"musicaudio.mp3")
        sound = AudioSegment.from_file(r"musicaudio.mp3")
        start_time = 0  
        end_time = 30 * 1000

        audio_segment = sound[start_time:end_time]

        audio_segment.export(r"music/downloaded/musicaudio.mp3", format="mp3")
        return True
    return False
    
def search_spotify(genres, tempo):
    SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
    SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

    client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
    spotifySearcher = spotipy.Spotify(client_credentials_manager=client_credentials_manager)



    seed_genres = [genres]
    target_tempo = int(tempo)
    min = target_tempo * 0.5
    max = target_tempo * 1.5

    print(seed_genres)
    recommendations = spotifySearcher.recommendations(seed_genres=seed_genres,  target_tempo=(min, max))

    if recommendations['tracks']:
        song = recommendations['tracks'][0]
        song_name = song['name']
        artist_name = song['artists'][0]['name']

        return f'Song: {song_name} by {artist_name}'
    else:
        return 'No recommendations found from spotify.'




matcher = Matcher(nlp.vocab)

patterns = [
    [{"LOWER": "hello"}],
    [{"LOWER": "hi"}],
    [{"LOWER": "how"}, {"LOWER": "are"}, {"LOWER": "you"}],
    [{"LOWER": "find"},{"LOWER": "me"},{"LOWER": "songs"},{"LOWER": "like"},{"LOWER": "this"}],
    [{"LOWER": "find"}, {"LOWER": "similiar"}, {"LOWER": "songs"}],
    [{"LOWER": "what"}, {"LOWER": "is"}],
    [{"LOWER": "increase"}, {"LOWER": "the"}, {"LOWER": {"REGEX": ".*"}}],
    [{"LOWER": "decrease"}, {"LOWER": "the"}, {"LOWER": {"REGEX": ".*"}}],

]

responses = {
    "greetings": [
        [{"LOWER": "hello"}],
        [{"LOWER": "hi"}],
    ],
    "inquiries": [
        [{"LOWER": "how"}, {"LOWER": "are"}, {"LOWER": "you"}],
    ],


    "find_change": [
    [{"LOWER": "increase"}, {"LOWER": "the"}, {"LOWER": {"REGEX": ".*"}}],
    [{"LOWER": "decrease"}, {"LOWER": "the"}, {"LOWER": {"REGEX": ".*"}}],

    ],
    "find_change_simple": [

    [{"LOWER": "make"}, {"LOWER": "it"}, {"LOWER": {"REGEX": ".*"}}],

    ],
 
    "predicitions": [
        [{"LOWER": "recos"}],
        [{"LOWER": "do"},{"LOWER": "you"},{"LOWER": "have"},{"LOWER": "a"},{"LOWER": "recommendation"},{"LOWER": "for"},{"LOWER": "me"}],

    ],

    "find_sim": [
        [{"LOWER": "search"},{"LOWER": "for"},{"LOWER": "similiar"},{"LOWER": "songs"}],
        [{"LOWER": "search"},{"LOWER": "for"},{"LOWER": "songs"},{"LOWER": "like"},{"LOWER": "this"}],
        [{"LOWER": "find"},{"LOWER": "me"},{"LOWER": "songs"},{"LOWER": "like"},{"LOWER": "this"}],

    ],

    "like": [
        [{"LOWER": "i"}, {"LOWER": "like"},  {"LOWER": {"REGEX": ".*"}}],
        [{"LOWER": "i"}, {"LOWER": "love"},  {"LOWER": {"REGEX": ".*"}}],
        [{"LOWER": "i"}, {"LOWER": "want"},  {"LOWER": {"REGEX": ".*"}}],
        [{"LOWER": "i"}, {"LOWER": "need"},  {"LOWER": {"REGEX": ".*"}}],
    ],

    "give_me": [
        [{"LOWER": "give"}, {"LOWER": "me"}, {"LOWER": "a"},  {"LOWER": {"REGEX": ".*"}}, {"LOWER": "song"},],

    ],

    "general": [
        [{"LOWER": "new"}],
        [{"LOWER": "bored"}],
        [{"LOWER": "what"},{"LOWER": "is"},{"LOWER": "your"},{"LOWER": "name"}],
        [{"LOWER": "do"},{"LOWER": "you"},{"LOWER": "like"}],
        [{"LOWER": "what"},{"LOWER": "is"},{"LOWER": {"REGEX": ".*"}}],
    ],    

}



for category, patterns in responses.items():
    for pattern in patterns:
        matcher.add(category, [pattern])


def general(user_input):
    newString = ""
    if user_input.find("name")!=-1:
        newString = "My name is DJ ORPHEUS, no need tell me yours"
    elif user_input.find("bored")!=-1:
        newString = "Thats actually crazy"
    elif user_input.find("like")!=-1:
        newString = "You don't want that answer lil bro"
    elif "pitch" in user_input or "chroma" in user_input:
        newString =  "Chroma or Pitch is represents the average pitch of the musical content. A value above 0.40 would be considered high"
    elif "harmony" in user_input:
        newString =  "Harmony represents the average harmonic component of an audio file. A value below 0.015 would be consdered low"
    elif "loudness" in user_input or "rms" in user_input:
        newString = " Loudness or RMS(Root Mean Sqaure) is a representation of the average amplitude of an audio signal. A value above 0.15 would be considered high"
    elif "energy" in user_input or "rolloff" in user_input:
        newString="Energy or Rolloff represents the average change in the frequency specified in percentage of an audio signal. A value above 4600 would be considered high"
    elif "sporadicty" in user_input or "sprectral_bandwith" in user_input:
        newString="Sporadicty or Spectral Bandwith represents the average width of an audio signal. A value below 2230 would be considered low"
    elif "brightness" in user_input or "sprectral_centroid" in user_input:
        newString="Brightness or Spectral Centroid represents the average of where the centroid of the spectrum is. A value above 2200 would be considered high"
    elif "tempo" in user_input:
        newString="Tempo represents the amount of beats per minute within an audio file."
    elif "beats" in user_input or "zero_crossing_rate" in user_input:
        newString=" Beats or Zero Crossing rate represents the average amount the signal of an audio file changes its sign. A value below 0.1 would be considered low"

    return newString

def give_me_a_song(user_input):
    newString = ""
    # target_name = ['blues', 'classical', 'country', 'disco', 'hiphop' ,'jazz' ,'metal', 'pop','reggae' ,'rock']
    if not [genre for genre in target_name if genre in user_input]:
        newString = "song must be real"
        return newString
    
    if "pop" in user_input:
        newString =  "Pop?? feeling a bit to upbeat?"
    elif "blues" in user_input:
        newString =  "Blues?? feeling a bit to sad?"
    elif "country" in user_input:
        newString =  "Country?? YE YE Heres a howdy song for you"
    elif "rock" in user_input:
        newString =  "Rock?? Rock on sibling"
    elif "hiphop" in user_input:
        newString =  "Hiphop?? let me put you on some bangers?"
    elif "jazz" in user_input:
        newString =  "Jazz?? Giant steps best jazz song, this probs not jazz"
    elif "metal" in user_input:
        newString =  "Metal?? AHHHHHHHHHHHHHHHHHHHHH!"
    elif "reggae" in user_input:
        newString =  "Reggae?? you are jamican me crazy.. that was bad sorry"
    elif "classical" in user_input:
        newString =  "Classical?? Amadeus ain't got nothing on me "
    elif "disco" in user_input:
        newString =  "Disco?? BOOGIE WONDERLAND" 

    newString = "give_me_a_song"+" " +newString  
    return newString


fast_words = ["faster","quicker"]
slow_words = ["slower","calmer"]
loud_words = ["louder","screamer"]
quiet_words = ["softer","quieter"]
darker_words = ["happier", "brighter"]
brighter_words = ["darker", "sadder"]

def chatbot_response(user_input, features1=None, userID=None):
    user_input = user_input.lower()
    doc = nlp(user_input)
    matches = matcher(doc)
    if matches:
        match_id, start,end = matches[0]
        category = nlp.vocab.strings[match_id]
        # print(f"Matched category: {category}, Span: {doc[start:end].text}")
        if category == "greetings":
            return "Hello! How can I assist you?",None,None,None,None
        elif category == "inquiries":
           return "I'm just the world's best DJ. How can I assist you?",None,None,None,None      
        elif category == "like":
            print("Loading....")  
            # extracted_word = doc[1].text
            before, keyword,extracted_word = doc.text.partition(doc[1].text)
            if(search(extracted_word)==True):
                features1 = extract_features(r"music/downloaded/musicaudio.mp3")
                genre1 = xgb.predict(features1)
                genreProb = xgb.predict_proba(features1)
                features1['filename'] = str(extracted_word)
                features1['label'] = genre1[0]
                label = label_encoder.inverse_transform(features1['label'])[0]
                high = confidence_score(genreProb)

                strLabel = "You " +keyword +" "+ str(extracted_word) + ". Based off the first 30 seconds of a song, it seems to be " +label+". Im saying that with "+ str(high[0][1])+"% confidence"
                return strLabel, None, features1,None, high
            else:
                strLabel = "Way to much content for me to process, narrow your keywords please"
                return strLabel, None, None,None, None
        elif category == "find_change":
            if features1 is None:
                strLabel=  "Exctract a song to use this great feature"
                return strLabel,None,None,None,None
            else:
                words = [token.text for token in doc if token.is_alpha]
                valid = False
                increaseVar = -1
                decreaseVar = -1
                input = user_input.split(" ")

                try:
                    increaseVar = input.index("increase")
                except ValueError:
                    pass
  
                try:
                    decreaseVar = input.index("decrease")
                except ValueError:
                    pass
                
                # print(increaseVar,":",decreaseVar)
                if(increaseVar > decreaseVar):
                    if(words[increaseVar+2] == "tempo"):
                        features = 'tempo'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(words[increaseVar+2] == "pitch" or words[increaseVar+2] == "chroma"):
                        features = 'chroma_stft_mean'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(words[increaseVar+2] == "harmony"):
                        features = 'harmony_mean'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(words[increaseVar+2] == "loudness" or words[increaseVar+2] == "rms"):
                        features = 'rms_mean'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(words[increaseVar+2] == "sporadcity" or words[increaseVar+2] == "bandwith"):
                        features = 'spectral_bandwidth_mean'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(words[increaseVar+2] == "brightness" or words[increaseVar+2] == "centroid"):
                        features = 'spectral_centroid_mean'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(words[increaseVar+2] == "beats" or words[increaseVar+2] == "crossingrate"):
                        features = 'zero_crossing_rate_mean'
                        value = features1[features]
                        value = value/100
                        valid=True
                else:
                    if(words[decreaseVar+2] == "tempo"):
                        features = 'tempo'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(words[decreaseVar+2] == "pitch" or words[decreaseVar+2] == "chroma"):
                        features = 'chroma_stft_mean'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(words[decreaseVar+2] == "harmony"):
                        features = 'harmony_mean'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(words[decreaseVar+2] == "loudness" or words[decreaseVar+2] == "rms"):
                        features = 'rms_mean'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(words[decreaseVar+2] == "sporadcity" or words[decreaseVar+2] == "bandwith"):
                        features = 'spectral_bandwidth_mean'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(words[decreaseVar+2] == "brightness" or words[decreaseVar+2] == "centroid"):
                        features = 'spectral_centroid_mean'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(words[decreaseVar+2] == "beats" or words[decreaseVar+2] == "crossingrate"):
                        features = 'zero_crossing_rate_mean'
                        value = features1[features]
                        value = -value/100
                        valid=True



                if(valid):
                    new_features = features1
                    new_features[features]+= value
                    songs = find_sim(new_features)
                    randomness = random.randint(0,9)
                    if(randomness>5):
                        return "Here are some of those increased features", songs, features1,None,None
                    else:
                        return "Here are some of those increased features, number 1 is my favourite", songs, features1,None,None
                else:
                    strLabel = "I'm sorry, but im going to need a valid song feature"
                    return strLabel, None,None,None,None            
        elif category == "find_change_simple":
            if features1 is None:
                strLabel=  "Exctract a song to use this great feature"
                return strLabel,None,None,None,None
            else:
                words = [token.text for token in doc if token.is_alpha]
                valid = False

                for s in words:
                    if(s in fast_words):
                        features = 'tempo'
                        value = features1[features]
                        value = value/100
                        valid=True
                    elif(s in slow_words):
                        features = 'tempo'
                        value = features1[features]
                        value = -value/100
                        valid=True
                    elif(s in loud_words):
                        features = 'rms_mean'
                        value =value/100
                        valid=True
                    elif(s in quiet_words):
                        features = 'rms_mean'
                        value = -value/100
                        valid=True
                    elif(s in brighter_words):
                        features = 'spectral_centroid_mean'
                        value = value/100
                        valid=True
                    elif(s in darker_words):
                        features = 'spectral_centroid_mean'
                        value = -value/100
                        valid=True
                
                if(valid):
                    new_features = features1
                    new_features[features]+= value
                    songs = find_sim(new_features)
                    randomness = random.randint(0,9)
                    if(randomness>5):
                        return "Here are some of those increased features", songs, features1,None,None
                    else:
                        return "Here are some of those increased features, number 1 is my favourite", songs, features1,None,None
                else:
                    strLabel = "I'm sorry, but im going to need a valid song feature"
                    return strLabel, None,None,None,None              
        elif category == "find_sim":      
            if features1 is None:
                strLabel=  "Exctract a song to use this great feature"
                return strLabel,None,None,None,None           
            else:
                sim = find_sim(features1)
                songs=[]
                #for key, value in sim.items():
                    #print(key," :",round(value,2),"% similiar")

                label = label_encoder.inverse_transform(features1['label'])[0]
                spotifySong = "Recommendation from Spotify: "+search_spotify(label,features1['tempo'])
                    
                return "Similiar Songs", sim,features1, spotifySong,None            
        elif category=="general":
            extracted_word = doc.text
            return general(extracted_word),None,None,None,None
        elif category=="give_me":
            extracted_word = doc.text
            info = give_me_a_song(extracted_word)
            return info,None,None,None,None
        elif category=="predicitions":
            firebase = firebase.FirebaseApplication('https://orpheus-3a4fa-default-rtdb.europe-west1.firebasedatabase.app/', None)
            result = firebase.get('/users', userID)
            pred= find_pred(result)

            if pred is None:
                return "Upload some more songs", None,None, None,None
            sim = find_sim(pred)
            # songs=[]   
            return "I have some songs that i think you might like", sim,pred, None,None          
    else:
        return "I'm sorry, I don't understand that, maybe type it a bit more clearer??.",None,None,None,None


def extract(name, filename):
    features1 = extract_features(name)

    genre1 = xgb.predict(features1)
    genreProb = xgb.predict_proba(features1)

    features1['filename'] = filename
    features1['label'] = genre1[0]
    label = label_encoder.inverse_transform(features1['label'])[0]
    high = confidence_score(genreProb)

    strLabel="This song is sounding a lot like the "+ label+" genre. Im saying with "+ str(high[0][1])+"% confidence"
    return features1,strLabel, high




app = Flask("chatterbot")
CORS(app) 


@app.route('/upload', methods=['POST'])
def upload():
    try:

        data = request.files['music_file']
        if(data is None):
            return jsonify({"status":"BAD REQUEST", "Orpheus":"Try better"}), 400

        name = "downloadedTest.mp3"
        data.save(name)
        features, response,high = extract(name, data.filename)
        features = features.to_json(orient='records')
        return jsonify({"status":"OK","Orpheus": response,"features":features, "confidence":high}), 200
    except Exception as e:
        return jsonify({"status": "Server Error", "Orpheus": "Uh oh"}), 500


@app.route('/chat', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        if(data.get('features')!=None):
            features1 = data.get('features')
            features1 = pd.read_json(features1)
        else:
            features1=None

        user_input = data.get('user_input')
        userID = data.get('userID')

        if(user_input is None or userID is None):
            return jsonify({"status":"BAD REQUEST", "Orpheus":"Try better"}), 400

        response,songs,features,recommendation, high = chatbot_response(user_input, features1, userID=userID)

        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            if(features.empty != True):
                features = features.to_json(orient='records')

        if isinstance(songs, pd.DataFrame) or isinstance(songs, pd.Series):
            if(songs.empty != True):
                songs = songs.to_json()

        return jsonify({"status":"OK","Orpheus": response,"songs":songs, "features": features,"recommendation": recommendation,"confidence":high }),200
    except Exception as e:
        print(e)
        return jsonify({"status": "Server Error", "Orpheus": "Uh oh"}), 500
if __name__ == '__main__':
    app.run(debug=True)