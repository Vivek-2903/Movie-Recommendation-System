import pandas as pd
import streamlit as st
import requests
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.card import card
from streamlit_extras.let_it_rain import rain

# --- CONFIG ---
st.set_page_config(
    page_title="🎬 MovieMagic Recommender",
    page_icon="🍿",
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    .movie-card {
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 20px;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    .movie-poster {
        border-radius: 10px;
        margin-bottom: 10px;
        width: 100%;
        height: 300px;
        object-fit: cover;
    }
    .match-score {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- IMAGE SETUP ---
MOVIE_THEMED_IMAGES = [
    "https://source.unsplash.com/random/300x450/?movie",
    "https://source.unsplash.com/random/300x450/?cinema",
    "https://source.unsplash.com/random/300x450/?film",
    "https://source.unsplash.com/random/300x450/?hollywood",
    "https://source.unsplash.com/random/300x450/?actor",
    "https://source.unsplash.com/random/300x450/?actress",
    "https://source.unsplash.com/random/300x450/?director",
    "https://source.unsplash.com/random/300x450/?oscar",
    "https://source.unsplash.com/random/300x450/?theater",
    "https://source.unsplash.com/random/300x450/?drama"
]

# --- TMDB API ---
TMDB_API_KEY = "YOUR_TMDB_API_KEY"

# --- FUNCTIONS ---
def load_data():
    df = pd.read_csv(r"C:\Users\vivek\Desktop\Movie Recommendation\data\movies.csv", encoding="latin1")
    df = df.drop(['duration', 'date_added', 'rating'], axis=1, errors='ignore')
    df = df.fillna({'director':'Unknown', 'cast':'Unknown', 'country':'Unknown'})
    df['features'] = df['title'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return df, cosine_sim, indices

def get_random_movie_image():
    """Returns a random movie-themed image URL"""
    return random.choice(MOVIE_THEMED_IMAGES)

def fetch_poster(title):
    """Try to get official poster, fallback to random image"""
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return get_random_movie_image()

def get_recommendations(title, df, cosine_sim, indices, n=10):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return df.iloc[movie_indices], [i[1] for i in sim_scores]
    except KeyError:
        return None, None

# --- MAIN APP ---
def main():
    df, cosine_sim, indices = load_data()
    
    with st.sidebar:
        st.title("🔍 Filters")
        all_genres = sorted(set(g.strip() for g in df['listed_in'].str.split(',').explode()))
        selected_genres = st.multiselect("Filter by genre", options=all_genres, default=[])
        if st.button("🎲 Get Random Movie"):
            random_movie = random.choice(df['title'].values)
            st.session_state.random_movie = random_movie
        rain(emoji="🎬", font_size=20)

    st.title("🍿 MovieMagic Recommender")
    movie_query = st.text_input(
        "Search for a movie...", 
        placeholder="The Dark Knight",
        value=st.session_state.get('random_movie', '')
    )

    if movie_query:
        try:
            searched_movie = df[df['title'] == movie_query].iloc[0]

            # Display searched movie details
            st.subheader(f"🎥 {searched_movie['title']}")
            col1, col2 = st.columns([1, 3])

            with col1:
                poster_url = fetch_poster(movie_query)
                if poster_url:
                    st.image(poster_url, use_column_width=True)

            with col2:
                st.write(f"**Director:** {searched_movie['director']}")
                st.write(f"**Cast:** {searched_movie['cast']}")
                st.write(f"**Country:** {searched_movie['country']}")
                st.write(f"**Genres:** {searched_movie['listed_in']}")
                st.write(f"**Description:** {searched_movie['description']}")

            st.divider()

            # Get and display recommendations
            recommendations, scores = get_recommendations(movie_query, df, cosine_sim, indices)

            if recommendations is not None and len(recommendations) > 0:
                if selected_genres:
                    mask = recommendations['listed_in'].apply(lambda x: any(g in x for g in selected_genres))
                    recommendations = recommendations[mask]
                    scores = [s for s, m in zip(scores, mask) if m]

                if len(recommendations) > 0:
                    st.subheader("🍿 Recommended Movies")
                    cols = st.columns(4)
                    
                    for i, (_, movie) in enumerate(recommendations.iterrows()):
                        with cols[i % 4]:
                            poster = fetch_poster(movie['title'])
                            score = scores[i]
                            
                            card_content = f"""
                            <div class="movie-card">
                                <img src="{poster}" class="movie-poster" alt="{movie['title']} poster">
                                <h4>{movie['title']}</h4>
                                <p>Similarity: <span class="match-score">{score*100:.0f}%</span></p>
                            </div>
                            """
                            st.markdown(card_content, unsafe_allow_html=True)
                else:
                    st.warning("No matching movies found with selected filters.")
            else:
                st.error("No recommendations found for this movie.")

        except IndexError:
            st.error("Movie not found in our database. Try another title!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
