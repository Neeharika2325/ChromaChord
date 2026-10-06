"""Module to map detected emotions to dynamic Telugu YouTube search queries."""

import random

# Mapping the 7 emotion classes to Telugu-specific search queries
MOOD_QUERIES = {
    "happy": [
        "happy telugu hit songs", 
        "telugu energetic dance songs", 
        "telugu blockbusters party anthems"
    ],
    "sad": [
        "telugu emotional sad songs", 
        "telugu heart touching melodies", 
        "telugu sad breakup songs"
    ],
    "neutral": [
        "telugu lofi beats", 
        "telugu calm melodies", 
        "telugu soothing relaxing songs"
    ],
    "surprise": [
        "telugu fast beat trending songs", 
        "telugu mass bgm", 
        "telugu upbeat trending tracks"
    ],
    "angry": [
        "telugu aggressive mass bgm", 
        "telugu high energy motivational songs", 
        "telugu hero elevation bgm"
    ],
    "disgust": [
        "telugu peaceful flute music", 
        "telugu classical fusion songs", 
        "telugu relaxing instrumental"
    ],
    "fear": [
        "telugu devotional songs", 
        "telugu calming bgm",
        "telugu soothing mantra music"
    ]
}

def get_search_query(emotion: str) -> str:
    """Returns a randomized Telugu search phrase based on the detected emotion."""
    # Convert emotion to lowercase to ensure it matches the dictionary keys
    emotion_key = emotion.lower()
    
    # Fallback to neutral if the emotion is somehow not recognized
    if emotion_key not in MOOD_QUERIES:
        emotion_key = "neutral"
        
    # Randomly select one of the queries from the list to keep results fresh
    return random.choice(MOOD_QUERIES[emotion_key])