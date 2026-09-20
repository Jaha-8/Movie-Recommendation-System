import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = ['Avatar', 'Titanic', 'Avengers', 'Interstellar']
genres = ['action sci-fi', 'romance drama', 'action hero', 'space sci-fi']

data = pd.DataFrame({
    'movie': movies,
    'genre': genres
})

cv = CountVectorizer()
matrix = cv.fit_transform(data['genre'])

similarity = cosine_similarity(matrix)

movie_name = input("Enter movie name: ")

if movie_name in data['movie'].values:

    index = data[data['movie'] == movie_name].index[0]

    scores = list(enumerate(similarity[index]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:\n")

    for movie in sorted_scores[1:]:
        print(data.iloc[movie[0]].movie)

else:
    print("Movie not found")