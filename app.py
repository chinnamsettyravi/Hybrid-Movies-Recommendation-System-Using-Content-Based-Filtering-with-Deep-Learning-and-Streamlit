import pickle
import streamlit as st
import requests
import os, json, hashlib, secrets, ast
from pathlib import Path
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd

# ---------------------- Simple user store ----------------------
USER_DB_PATH = Path("users_db.json")

def _load_user_db():
    if USER_DB_PATH.exists():
        try:
            with open(USER_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_user_db(db: dict):
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def register_user(username: str, password: str):
    db = _load_user_db()
    if username in db:
        return False, "Username already exists."
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    db[username] = {"salt": salt, "pwd_hash": pwd_hash}
    _save_user_db(db)
    return True, "Signup successful! Please log in."

def verify_login(username: str, password: str):
    db = _load_user_db()
    user = db.get(username)
    if not user:
        return False, "User not found."
    if _hash_password(password, user["salt"]) != user["pwd_hash"]:
        return False, "Incorrect password."
    return True, "Login successful."

# ---------------------- Poster fetch ----------------------
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"

    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return None
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return full_path
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster for movie ID {movie_id}: {e}")
        return None

# ---------------------- Recommendation (uses your model) ----------------------
def recommend(movie):
    # Uses the global 'movies' and 'similarity' loaded from your pickles
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        row = movies.iloc[i[0]]
        movie_id = row['movie_id'] if 'movie_id' in row else row.get('id', None)
        poster = fetch_poster(movie_id) if movie_id is not None else None
        recommended_movie_posters.append(poster)
        recommended_movie_names.append(row['title'] if poster else f"{row['title']} (Poster not available)")
    return recommended_movie_names, recommended_movie_posters

# ---------------------- Auth UI ----------------------
def auth_ui():
    st.title("🔐 Welcome")
    st.write("Please **Sign Up** first, then **Log In** to use the app.")

    tabs = st.tabs(["📝 Sign Up", "🔑 Log In"])

    # --- Sign Up ---
    with tabs[0]:
        su_user = st.text_input("Username", key="su_user")
        su_pass = st.text_input("Password", type="password", key="su_pass")
        su_btn = st.button("Create Account")
        if su_btn:
            ok, msg = register_user(su_user.strip(), su_pass)
            if ok:
                st.success(msg)
                st.session_state.show_login = True
            else:
                st.error(msg)

    # --- Log In ---
    with tabs[1]:
        db_empty = len(_load_user_db()) == 0
        if db_empty:
            st.info("No accounts yet. Please sign up first.")
        li_user = st.text_input("Username", key="li_user", disabled=db_empty)
        li_pass = st.text_input("Password", type="password", key="li_pass", disabled=db_empty)
        li_btn = st.button("Log In", disabled=db_empty)
        if li_btn and not db_empty:
            ok, msg = verify_login(li_user.strip(), li_pass)
            if ok:
                st.success(msg)
                st.session_state.logged_in = True
                st.session_state.username = li_user.strip()
                st.session_state.show_landing = True  # go to landing first
                st.rerun()
            else:
                st.error(msg)

# ---------------------- App state ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "show_landing" not in st.session_state:
    st.session_state.show_landing = False

# ---------------------- Load raw TMDB CSVs for filters ----------------------
@st.cache_data(show_spinner=False)
def load_raw_dataset(
    movies_csv_path: str = "tmdb_5000_movies.csv",
    credits_csv_path: str = "tmdb_5000_credits.csv"
) -> pd.DataFrame:
    """
    Loads raw TMDB datasets and returns a merged DataFrame with:
      - title, id (movie id), genre_list (list[str]), cast_list (list[str])
      - popularity, release_date (for sorting)
    """
    try:
        movies_df = pd.read_csv(movies_csv_path)
    except Exception as e:
        st.error(f"Error loading {movies_csv_path}: {e}")
        return pd.DataFrame()

    # credits is optional but recommended to get real cast names (heroes)
    cast_df = None
    try:
        credits_df = pd.read_csv(credits_csv_path)
        # credits: id (movie id), title, cast (JSON), crew (JSON)
        cast_df = credits_df[["movie_id", "title", "cast"]].rename(columns={"movie_id": "id"})
    except Exception:
        cast_df = None

    # helpers to parse json-like strings
    def safe_list_names(s):
        try:
            arr = ast.literal_eval(s)
            return [d.get("name", "").strip() for d in arr if isinstance(d, dict) and d.get("name")]
        except Exception:
            return []

    movies_df["genre_list"] = movies_df.get("genres", "").apply(safe_list_names)
    movies_df["id"] = movies_df["id"]  # ensure id present

    if cast_df is not None:
        cast_df["cast_list"] = cast_df["cast"].apply(safe_list_names)
        merged = movies_df.merge(cast_df[["id", "cast_list"]], on="id", how="left")
    else:
        # Fall back: if credits file missing, approximate heroes via keywords (not ideal)
        movies_df["cast_list"] = movies_df.get("keywords", "").apply(safe_list_names)
        merged = movies_df

    # Keep the useful columns
    keep_cols = ["title", "id", "genre_list", "cast_list", "popularity", "release_date"]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[keep_cols].copy()

raw_movies = load_raw_dataset()

# ---------------------- Landing Page ----------------------
def landing_page():
    st.title("🎥 Movie Recommender System")
    st.subheader("About this project")
    st.write("""
This project is a **Content-based Movie Recommendation System**.

**Core idea — Cosine Similarity:**  
We represent each movie as a vector of its features (e.g., genres, keywords, cast, crew).  
The **cosine similarity** between two vectors measures how close their directions are (irrespective of magnitude).  
- If two movies share many similar features → their vectors point in a similar direction → **high cosine similarity** → they get recommended together.

**How it works in this app:**  
1. You first filter by **Genre** or **Hero (cast)**.  
2. We fetch **top 5 matching movies** directly from the raw TMDB dataset (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`).  
3. You select one of these movies.  
4. Then we use your precomputed model (**`movie_list.pkl` + `similarity.pkl`**) to get the **top 5 similar movies** via cosine similarity.  
    """)

    if st.button("🚀 Start Using App", use_container_width=True):
        st.session_state.show_landing = False
        st.rerun()

# ---------------------- Main ----------------------
if not st.session_state.logged_in:
    auth_ui()
else:
    if st.session_state.show_landing:
        landing_page()
    else:
        left, right = st.columns([1, 1])
        with left:
            st.header('🎬 Movie Recommender System')
            st.caption(f"Logged in as **{st.session_state.username}**")
        with right:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.show_landing = False
                st.rerun()

        # -------- Load the movies and similarity data (YOUR original models) --------
        movies, similarity = None, None
        load_errors = []

        try:
            movies = pickle.load(open(r'C:\Users\vekkudu naveen\OneDrive\Desktop\machine-learning-projects\movie-recommender-system\movie_list.pkl', 'rb'))
        except Exception as e:
            load_errors.append(str(e))
            try:
                movies = pickle.load(open('movie_list.pkl', 'rb'))
            except Exception as e2:
                load_errors.append(str(e2))

        try:
            similarity = pickle.load(open(r'C:\Users\vekkudu naveen\OneDrive\Desktop\machine-learning-projects\movie-recommender-system\similarity.pkl', 'rb'))
        except Exception as e:
            load_errors.append(str(e))
            try:
                similarity = pickle.load(open('similarity.pkl', 'rb'))
            except Exception as e2:
                load_errors.append(str(e2))

        if movies is None or similarity is None:
            st.error("Could not load model files (`movie_list.pkl` / `similarity.pkl`). Please check the file paths.")
            if load_errors:
                with st.expander("Show load errors"):
                    for err in load_errors:
                        st.code(err)
            st.stop()

        # --------- NEW: Filters powered by raw TMDB CSVs (avoids KeyError on 'genres') ---------
        if raw_movies is None or raw_movies.empty:
            st.warning("Raw dataset not found or empty. Please place tmdb_5000_movies.csv (and optionally tmdb_5000_credits.csv) next to this script.")
            st.stop()

        # Build the unique choices
        all_genres = sorted({g for sub in raw_movies["genre_list"].dropna() for g in (sub or [])})
        all_heroes = sorted({c for sub in raw_movies["cast_list"].dropna() for c in (sub or [])})

        st.subheader("🔎 Filter first")
        c1, c2 = st.columns(2)
        with c1:
            selected_genres = st.multiselect("🎭 Select Genre(s)", all_genres, default=[])
        with c2:
            selected_heroes = st.multiselect("🦸 Select Hero/Actor(s)", all_heroes, default=[])

        # Apply filters
        filtered = raw_movies.copy()

        if selected_genres:
            filtered = filtered[filtered["genre_list"].apply(lambda gl: bool(set(gl or []) & set(selected_genres)))]
        if selected_heroes:
            filtered = filtered[filtered["cast_list"].apply(lambda cl: bool(set(cl or []) & set(selected_heroes)))]

        # If nothing selected, gently nudge user
        if not selected_genres and not selected_heroes:
            st.info("Tip: Pick at least a **Genre** or a **Hero** to see matching movies.")
        
        # Sort to show sensible "top" rows (by popularity, then recent release)
        try:
            filtered["release_date"] = pd.to_datetime(filtered["release_date"], errors="coerce")
        except Exception:
            pass
        filtered = filtered.sort_values(by=["popularity", "release_date"], ascending=[False, False])

        # Show top 5 candidates from the filter
        st.subheader("📌 Top 5 movies from your selection")
        top5 = filtered.head(5)
        if top5.empty:
            st.warning("No movies match your filters. Try different selections.")
            st.stop()

        # Display a compact list
        for _, row in top5.iterrows():
            genres_str = ", ".join(row["genre_list"] or [])
            cast_preview = ", ".join((row["cast_list"] or [])[:4])
            st.write(f"• **{row['title']}** — _Genres:_ {genres_str}  |  _Cast:_ {cast_preview}")

        # Now the user selects ONE of these movies, then we call your recommender
        selected_movie = st.selectbox("🎬 Now select a movie to get recommendations", top5["title"].values)

        # Safety: if title not found in your 'movies' pickle (sometimes names differ), try case-insensitive match
        def resolve_title_in_model(title: str) -> str | None:
            if title in movies["title"].values:
                return title
            # fallback case-insensitive
            lower_map = {t.lower(): t for t in movies["title"].values}
            return lower_map.get(title.lower(), None)

        if st.button('Show Recommendation'):
            resolved = resolve_title_in_model(selected_movie)
            if not resolved:
                st.error(f"'{selected_movie}' is not present in the model index. Please pick another movie from the list.")
            else:
                try:
                    recommended_movie_names, recommended_movie_posters = recommend(resolved)
                    cols = st.columns(5)
                    for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
                        col.text(name)
                        if poster:
                            col.image(poster)
                        else:
                            col.caption("Poster not available")
                except Exception as e:
                    st.error(f"Could not generate recommendations: {e}")
