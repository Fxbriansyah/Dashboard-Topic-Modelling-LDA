import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora
import gensim
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.stats import chi2_contingency
from plotly.subplots import make_subplots
import google.generativeai as genai

# Initialize session state for persisting values between reruns
if 'lda_params' not in st.session_state:
    st.session_state.lda_params = {
        'num_topics': 3,
        'passes': 15,
        'chunksize': 100,
        'update_every': 1,
        'alpha': 'auto',
        'eta': 'auto',
        'random_state': 42,
        'run_model': False
    }

# Initialize Gemini session state
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
    st.session_state.gemini_model = None

# Set page config
st.set_page_config(
    page_title="Dashboard LDA - Review Konsumen Tokopedia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #42b883;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #35495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f5f5f5;
        margin-bottom: 1rem;
    }
    .ai-response {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #42b883;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>📊 Dashboard Analisis Topik Review Konsumen Tokopedia</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class='card'>
    <p>Dashboard ini digunakan untuk melakukan <em>topic modeling</em> menggunakan LDA 
    (Latent Dirichlet Allocation) pada ulasan konsumen di Tokopedia. Analisis ini membantu memahami 
    tema-tema utama dalam ulasan pelanggan dengan dukungan AI dan analisis asosiasi kata.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ PREPROCESSING FUNCTIONS ------------------

def clean_text(text):
    """Clean text from non-alphabetic characters and remove excess spaces."""
    if isinstance(text, str):
        # Lowercase
        text = text.lower()
        # Hapus karakter non-alfabet kecuali spasi
        text = re.sub(r'[^a-z\s]', '', text)
        # Kurangi karakter berulang (contoh: mantaaap => mantap)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        # Normalisasi spasi
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def case_folding(text):
    """Convert text to lowercase."""
    return text.lower() if isinstance(text, str) else ""

def tokenize_text(text):
    """Tokenize text into words."""
    if isinstance(text, str):
        return word_tokenize(text)
    return []

# Create stemmer for Indonesian
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_words(tokens):
    """Stem tokenized words."""
    return [stemmer.stem(word) for word in tokens if isinstance(word, str)]

custom_stopwords = [
    "bgt", "gk", "ga", "nggak", "gak", "dpt", "tp", "dr", "sy", "sm", "cmn", "trs", "sip",
    "tuh", "aja", "kok", "sih", "nih", "yah", "deh", "dah", "udh", "lg", "krn", "sj", "k",
    "y", "mantap", "mantab", "kasih", "josss", "bagus", "hr", "com", "sat", "set", "mksh",
    "hd", "yg", "ya", "cepat", "aman", "lumayan", "fast", "respon", "dgn", "sellerbarang",
    "nya", "dipacking", "mimin", "kuliahkantor", "tugas", "barang", "produk", "seller",
    "kirim", "paket", "deskripsi", "fungsi", "proses", "warna", "bonus", "harap", "admin",
    "oke", "jg", "ok", "top", "keren", "desain", "kenceng", "makasih", "spek", "mouse",
    "response", "harga", "kualitas", "responsif", "terima", "gercep", "kurir", "bantu",
    "sesuai", "ramah", "layan", "jual", "pesan", "banget", "packaging", "packing", "suara",
    "brg", "langsung", "order", "kk", "pake", "sampe", "rb", "gan", "n", "sdh", "beli", "toko",
    "puas", "service", "baik", "adminnya", "minggu", "terimakasih", "responnya",
    "request", "rekom", "asik", "sayang", "jos", "min", "orang", "pegawai", "alhamdulillah",
    "mantappp", "mantull", "baikthx", "dapat", "cek", "review", "repeat", "transaksi",
    "alamat", "sukses", "thanks", "recomended", "kerja", "online", "store", "promo","nya"
    "ambil", "komunikatif", "saing"
]

# Get stopwords for Indonesian and English
stop_words = stopwords.words('indonesian') + stopwords.words('english')

def remove_stopwords(tokens, custom_stopwords=[]):
    """Remove stopwords from tokenized text."""
    all_stopwords = set(stop_words + custom_stopwords)
    return [word for word in tokens if word not in all_stopwords and len(word) > 1]

# ------------------ BIGRAM FUNCTIONS ------------------

def create_bigrams(texts, min_count=5, threshold=100):
    """Create bigrams from tokenized texts."""
    # Create bigram model
    bigram = Phrases(texts, min_count=min_count, threshold=threshold)
    bigram_mod = Phraser(bigram)
    
    # Apply bigram model to texts
    return [bigram_mod[doc] for doc in texts]

def visualize_bigrams(texts, top_n=20):
    """Visualize most common bigrams."""
    # Extract bigrams
    bigram_list = []
    for text in texts:
        for i in range(len(text) - 1):
            bigram = f"{text[i]}_{text[i+1]}"
            bigram_list.append(bigram)
    
    # Count bigrams
    bigram_counts = Counter(bigram_list)
    top_bigrams = bigram_counts.most_common(top_n)
    
    # Create DataFrame
    df_bigrams = pd.DataFrame(top_bigrams, columns=['Bigram', 'Count'])
    
    # Create bar chart
    fig = px.bar(df_bigrams, 
                 x='Count', 
                 y='Bigram', 
                 orientation='h',
                 title=f'Top {top_n} Bigrams',
                 labels={'Count': 'Frekuensi', 'Bigram': 'Bigram'})
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    return df_bigrams

# ------------------ WORD ASSOCIATION FUNCTIONS ------------------

def calculate_word_associations(texts, target_word, top_n=10):
    """Calculate word associations using co-occurrence matrix."""
    # Create co-occurrence matrix
    vectorizer = CountVectorizer(ngram_range=(1, 1))
    
    # Convert texts to string format
    text_strings = [' '.join(text) for text in texts]
    X = vectorizer.fit_transform(text_strings)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Find target word index
    if target_word not in feature_names:
        st.warning(f"Kata '{target_word}' tidak ditemukan dalam korpus.")
        return pd.DataFrame()
    
    target_idx = list(feature_names).index(target_word)
    
    # Calculate co-occurrence
    cooccurrence = X.T.dot(X)
    target_cooccurrence = cooccurrence[target_idx].toarray().flatten()
    
    # Create association scores
    word_associations = []
    for i, word in enumerate(feature_names):
        if i != target_idx and target_cooccurrence[i] > 0:
            # Calculate association strength (normalized co-occurrence)
            association_score = target_cooccurrence[i] / (X[:, i].sum() + X[:, target_idx].sum() - target_cooccurrence[i])
            word_associations.append((word, target_cooccurrence[i], association_score))
    
    # Sort by association score
    word_associations.sort(key=lambda x: x[2], reverse=True)
    
    # Create DataFrame
    df_associations = pd.DataFrame(word_associations[:top_n], 
                                   columns=['Kata', 'Co-occurrence', 'Association Score'])
    
    return df_associations

def visualize_word_network(texts, min_cooccurrence=3, top_words=20):
    """Create word association network visualization."""
    # Create co-occurrence matrix
    vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=top_words)
    text_strings = [' '.join(text) for text in texts]
    X = vectorizer.fit_transform(text_strings)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate co-occurrence matrix
    cooccurrence = X.T.dot(X)
    cooccurrence.setdiag(0)  # Remove self-connections
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for word in feature_names:
        G.add_node(word)
    
    # Add edges based on co-occurrence
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            weight = cooccurrence[i, j]
            if weight >= min_cooccurrence:
                G.add_edge(feature_names[i], feature_names[j], weight=weight)
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract edges
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_info.append(f"{edge[0]} - {edge[1]}: {G.edges[edge]['weight']}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Calculate node degree
        adjacencies = list(G.neighbors(node))
        node_info.append(f"{node}<br>Connections: {len(adjacencies)}")
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=10,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                xanchor="left",
                title=dict(side="right")
            )
        )
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='Word Association Network',
                font=dict(size=16)  # Format baru untuk title font
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="Ukuran node menunjukkan jumlah koneksi",
                showarrow=False,
                xref="paper", 
                yref="paper",
                x=0.005, 
                y=-0.002,
                xanchor='left', 
                yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ------------------ GEMINI AI FUNCTIONS ------------------

def setup_gemini_client(api_key=None):
    """Setup Gemini client dengan API key"""
    api_key = api_key or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.warning("⚠️ Gemini API key belum diatur. Masukkan API key di sidebar untuk menggunakan fitur AI.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def generate_topic_insights(topics_data, reviews_sample):
    """Generate topic analysis insights"""
    if not topics_data or not reviews_sample:
        st.warning("Data topik atau ulasan tidak valid")
        return None
        
    try:
        model = setup_gemini_client(st.session_state.get("gemini_api_key"))
        if not model:
            return None
            
        topics_text = "\n".join([f"Topik {i+1}: {topic['Kata-kata']}" for i, topic in enumerate(topics_data)])
        sample_text = "\n".join(reviews_sample[:5])
        
        prompt = f"""
Berdasarkan data berikut, berikan analisis mendalam dalam Bahasa Indonesia:

**Topik yang Ditemukan:**
{topics_text}

**Contoh Ulasan:**
{sample_text}

**Tugas:**
1. Interpretasikan setiap topik dalam konteks bisnis e-commerce
2. Berikan insight tentang kepuasan pelanggan
3. Rekomendasi tindakan perbaikan
4. Identifikasi pola penting

Format output:
- Gunakan bahasa profesional
- Poin-poin spesifik
- Maksimal 150 kata
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

def generate_word_association_insights(associations_data, target_word):
    """Generate insights for word associations"""
    if associations_data.empty:
        st.warning("Data asosiasi kosong")
        return None
        
    try:
        model = setup_gemini_client(st.session_state.get("gemini_api_key"))
        if not model:
            return None
            
        associations_text = "\n".join(
            f"- {row['Kata']} (Score: {row['Association Score']:.3f})"
            for _, row in associations_data.iterrows()
        )
        
        prompt = f"""
Analisis asosiasi kata untuk '{target_word}':

**Data Asosiasi:**
{associations_text}

**Tugas:**
1. Jelaskan makna asosiasi ini dalam konteks e-commerce
2. Interpretasi persepsi pelanggan
3. Rekomendasi bisnis spesifik
4. Saran perbaikan produk/layanan

Format:
- Gunakan Bahasa Indonesia
- Poin-poin jelas
- Maksimal 100 kata
"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        st.error(f"Error generating association insights: {str(e)}")
        return None

# ------------------ VISUALIZATION FUNCTIONS ------------------

def create_wordcloud(text_list, title="Word Cloud"):
    """Create and display word cloud from preprocessed text."""
    all_words = ' '.join([' '.join(words) for words in text_list])
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=150,
        contour_width=3,
        contour_color='steelblue',
        collocations=False
    ).generate(all_words)
    
    # Display
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=15)
    ax.axis('off')
    st.pyplot(fig)

def plot_topic_wordclouds(lda_model, num_topics):
    """Create word clouds for each topic."""
    # Create a figure with subplots
    fig, axes = plt.subplots(int(np.ceil(num_topics/2)), 2, figsize=(15, num_topics*2))
    axes = axes.flatten()
    
    # Generate word clouds for each topic
    for i, topic_id in enumerate(range(num_topics)):
        # Get the top 30 words for the topic
        topic_words = dict(lda_model.show_topic(topic_id, 30))
        
        # Create a word cloud
        wc = WordCloud(
            background_color='white',
            width=800,
            height=400,
            max_words=50,
            prefer_horizontal=1.0,
            colormap='viridis',
            contour_width=2,
            contour_color='steelblue'
        )
        
        # Generate the word cloud from the topic words
        cloud = wc.generate_from_frequencies(topic_words)
        
        # Plot the word cloud on the axis
        axes[i].imshow(cloud, interpolation='bilinear')
        axes[i].set_title(f'Topic {i+1}', fontsize=14)
        axes[i].axis('off')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_topics_distribution(corpus, lda_model):
    """Plot the distribution of topics across documents with a simple chart."""
    # Get topic distribution for each document
    topic_distribution = []
    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        # Convert to a full distribution (all topics)
        topic_dist = [0] * lda_model.num_topics
        for topic_id, prob in topics:
            topic_dist[topic_id] = prob
        topic_distribution.append(topic_dist)
    
    # Convert to DataFrame
    topic_df = pd.DataFrame(topic_distribution)
    topic_df.columns = [f"Topic {i+1}" for i in range(lda_model.num_topics)]
    
    # Calculate average topic proportion
    avg_topic_props = topic_df.mean().reset_index()
    avg_topic_props.columns = ['Topic', 'Average Proportion']
    
    # Create simple bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(avg_topic_props['Topic'], avg_topic_props['Average Proportion'], color='steelblue')
    
    # Add labels and title
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Average Proportion', fontsize=12)
    ax.set_title('Average Topic Distribution Across Documents', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Also display as a table for clarity
    st.markdown("#### Distribusi Rata-rata Topik")
    st.dataframe(avg_topic_props.set_index('Topic').style.format({"Average Proportion": "{:.4f}"}),
                 use_container_width=True)

def plot_topic_dominance(corpus, lda_model):
    """Plot pie chart showing dominant topics in corpus."""
    # Determine dominant topic for each document
    dominant_topics = []
    for i, bow in enumerate(corpus):
        topics = lda_model.get_document_topics(bow)
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)
    
    # Count occurrences of each dominant topic
    topic_counts = Counter(dominant_topics)
    
    # Prepare data for pie chart
    labels = [f"Topic {i+1}" for i in range(lda_model.num_topics)]
    values = [topic_counts.get(i, 0) for i in range(lda_model.num_topics)]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(colors=px.colors.qualitative.Bold)
    )])
    
    fig.update_layout(title_text='Dominant Topics Distribution')
    st.plotly_chart(fig, use_container_width=True)

def evaluate_coherence_values(texts, dictionary, corpus, start=2, stop=12, step=1):
    """
    Compute coherence values for different numbers of topics
    """
    coherence_values = []
    model_list = []
    topic_nums = range(start, stop, step)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, num_topics in enumerate(topic_nums):
        status_text.text(f'Building LDA model with {num_topics} topics...')
        model = gensim.models.LdaModel(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=num_topics, 
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto'
        )
        model_list.append(model)
        
        status_text.text(f'Calculating coherence for {num_topics} topics...')
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        
        # Update progress
        progress_bar.progress((i + 1) / len(topic_nums))
    
    status_text.text('Done!')
    progress_bar.empty()
    
    # Plot coherence values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(topic_nums, coherence_values)
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score")
    ax.set_title("Coherence Scores by Number of Topics")
    ax.set_xticks(topic_nums)
    ax.grid(True)
    st.pyplot(fig)
    
    return model_list, coherence_values

def create_pyldavis_visualization(lda_model, corpus, dictionary):
    """Create pyLDAvis visualization."""
    # Prepare visualization
    try:
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        
        # Save to HTML file
        html_string = pyLDAvis.prepared_data_to_html(vis_data)
        
        # Display in Streamlit
        st.components.v1.html(html_string, width=1300, height=800)
    except Exception as e:
        st.error(f"Error creating LDAvis visualization: {str(e)}")

# ------------------ MAIN APP LOGIC ------------------

# Sidebar
st.sidebar.markdown("## 📄 Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV/Excel", type=["csv", "xlsx"])

# Gemini API Key input
st.sidebar.markdown("## 🤖 Konfigurasi Gemini")
api_key_input = st.sidebar.text_input(
    "Gemini API Key", 
    type="password", 
    value=st.session_state.get("gemini_api_key", ""),
    help="Dapatkan di https://aistudio.google.com"
)

if api_key_input:
    st.session_state.gemini_api_key = api_key_input
    try:
        genai.configure(api_key=api_key_input)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        st.sidebar.success("✅ API Key valid!")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

# Custom stopwords input
st.sidebar.markdown("## 🔍 Preprocessing")
custom_stop_input = st.sidebar.text_area("Custom Stopwords (pisahkan dengan koma)", "")
user_custom_stopwords = [word.strip().lower() for word in custom_stop_input.split(',') if word.strip()]

# Bigram settings
st.sidebar.markdown("## 🔗 Bigram Settings")
min_count = st.sidebar.slider("Min Count", 1, 20, 5)
threshold = st.sidebar.slider("Threshold", 0, 100, 10)

# Check if file is uploaded
if uploaded_file is not None:
    # Read uploaded file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File berhasil diunggah! {df.shape[0]} baris dan {df.shape[1]} kolom.")
        
        # Display dataframe sample
        with st.expander("Preview Data"):
            st.dataframe(df.head())
        
        # Select column for analysis
        text_col = st.selectbox(
            "Pilih kolom yang berisi ulasan untuk dianalisis:",
            options=df.columns.tolist(),
            index=df.columns.get_loc('ulasan') if 'ulasan' in df.columns else 0
        )
        
        # Preprocessing
        st.markdown("<h2 class='sub-header'>🔄 Preprocessing Data</h2>", unsafe_allow_html=True)
        
        # Menghapus duplikat berdasarkan kolom ulasan
        df = df.drop_duplicates(subset=text_col, keep='first')
        
        with st.spinner("Melakukan preprocessing..."):
            # 1. Cleaning text
            df['cleaning_text'] = df[text_col].apply(clean_text)
            
            # 2. Case folding
            df['case_folding'] = df['cleaning_text'].apply(case_folding)
            
            # 3. Tokenization
            df['tokenisasi'] = df['case_folding'].apply(tokenize_text)
            
            # 4. Stemming
            df['stemming'] = df['tokenisasi'].apply(stem_words)
            
            # 5. Stopword removal
            df['stopwords'] = df['stemming'].apply(lambda x: remove_stopwords(x, user_custom_stopwords))
            
            # 6. Create bigrams
            df['ngram_tokens'] = create_bigrams(df['stopwords'].tolist(), min_count=min_count, threshold=threshold)
            
            # Filter rows with empty tokens
            df_filtered = df[df['ngram_tokens'].map(len) > 0].copy()
            
            st.success(f"✅ Preprocessing selesai! Tersisa {df_filtered.shape[0]} dokumen setelah filtering.")

        # Display preprocessing results
        with st.expander("Hasil Preprocessing"):
            st.dataframe(df_filtered[[text_col, 'cleaning_text', 'case_folding', 'tokenisasi', 'stemming', 'stopwords', 'ngram_tokens']])
        
        # Word Cloud dari hasil preprocessing
        st.markdown("<h2 class='sub-header'>☁️ Word Cloud</h2>", unsafe_allow_html=True)
        create_wordcloud(df_filtered['ngram_tokens'].tolist(), "Word Cloud dari Preprocessing")
        
        # Visualisasi Bigrams
        st.markdown("<h2 class='sub-header'>🔗 Analisis Bigrams</h2>", unsafe_allow_html=True)
        bigram_df = visualize_bigrams(df_filtered['ngram_tokens'].tolist())
        
        # Word Association Analysis
        st.markdown("<h2 class='sub-header'>🔍 Analisis Asosiasi Kata</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            target_word = st.text_input("Masukkan kata untuk analisis asosiasi:", "")
        with col2:
            top_n_assoc = st.slider("Jumlah asosiasi teratas:", 5, 20, 10)
        
        if target_word:
            associations_df = calculate_word_associations(df_filtered['ngram_tokens'].tolist(), target_word, top_n_assoc)
            
            if not associations_df.empty:
                st.dataframe(associations_df, use_container_width=True)
                
                # AI Insights for word associations
                if st.session_state.get("gemini_api_key"):
                    with st.spinner("Generating AI insights for word associations..."):
                        word_insights = generate_word_association_insights(associations_df, target_word)
                        if word_insights:
                            st.markdown("<div class='ai-response'>", unsafe_allow_html=True)
                            st.markdown("### 🤖 AI Insights - Asosiasi Kata")
                            st.write(word_insights)
                            st.markdown("</div>", unsafe_allow_html=True)
        
        # Word Network Visualization
        st.markdown("<h2 class='sub-header'>🌐 Network Asosiasi Kata</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            min_cooccurrence = st.slider("Minimum co-occurrence:", 2, 10, 3)
        with col2:
            top_words_network = st.slider("Jumlah kata teratas:", 10, 50, 20)
        
        if st.button("Generate Word Network"):
            with st.spinner("Creating word network..."):
                visualize_word_network(df_filtered['ngram_tokens'].tolist(), min_cooccurrence, top_words_network)
        
        # LDA Topic Modeling
        st.markdown("<h2 class='sub-header'>📊 LDA Topic Modeling</h2>", unsafe_allow_html=True)
        
        # LDA Parameters
        st.markdown("### ⚙️ Parameter LDA")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.lda_params['num_topics'] = st.slider("Jumlah Topik", 2, 15, st.session_state.lda_params['num_topics'])
            st.session_state.lda_params['passes'] = st.slider("Passes", 5, 50, st.session_state.lda_params['passes'])
        
        with col2:
            st.session_state.lda_params['chunksize'] = st.slider("Chunk Size", 50, 500, st.session_state.lda_params['chunksize'])
            st.session_state.lda_params['update_every'] = st.slider("Update Every", 1, 10, st.session_state.lda_params['update_every'])
        
        with col3:
            alpha_options = ['auto', 'symmetric', 'asymmetric']
            st.session_state.lda_params['alpha'] = st.selectbox("Alpha", alpha_options, index=alpha_options.index(st.session_state.lda_params['alpha']))
            eta_options = ['auto', 'symmetric']
            st.session_state.lda_params['eta'] = st.selectbox("Eta", eta_options, index=eta_options.index(st.session_state.lda_params['eta']))
        
        # Button to run LDA
        if st.button("🚀 Jalankan LDA Topic Modeling"):
            st.session_state.lda_params['run_model'] = True
        
        # Run LDA if button is clicked
        if st.session_state.lda_params['run_model']:
            with st.spinner("Melakukan LDA Topic Modeling..."):
                # Prepare data for LDA
                processed_docs = df_filtered['ngram_tokens'].tolist()
                
                # Create dictionary and corpus
                dictionary = corpora.Dictionary(processed_docs)
                dictionary.filter_extremes(no_below=2, no_above=0.8)
                corpus = [dictionary.doc2bow(text) for text in processed_docs]
                
                # Build LDA model
                lda_model = gensim.models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=st.session_state.lda_params['num_topics'],
                    random_state=st.session_state.lda_params['random_state'],
                    passes=st.session_state.lda_params['passes'],
                    chunksize=st.session_state.lda_params['chunksize'],
                    update_every=st.session_state.lda_params['update_every'],
                    alpha=st.session_state.lda_params['alpha'],
                    eta=st.session_state.lda_params['eta'],
                    per_word_topics=True
                )
                
                # Store results in session state
                st.session_state.lda_model = lda_model
                st.session_state.dictionary = dictionary
                st.session_state.corpus = corpus
                st.session_state.processed_docs = processed_docs
            
            st.success("✅ LDA Topic Modeling berhasil!")
            
            # Display topics
            st.markdown("### 📋 Topik yang Ditemukan")
            topics_data = []
            for i, topic in enumerate(lda_model.print_topics()):
                topic_words = [word.split('*')[1].strip().replace('"', '') for word in topic[1].split(' + ')]
                topics_data.append({
                    'Topik': f'Topik {i+1}',
                    'Kata-kata': ', '.join(topic_words[:10]),
                    'Detail': topic[1]
                })
            
            topics_df = pd.DataFrame(topics_data)
            st.dataframe(topics_df[['Topik', 'Kata-kata']], use_container_width=True)
            
            # Store topics data in session state for AI insights
            st.session_state.topics_data = topics_data
            
            # Topic Word Clouds
            st.markdown("### ☁️ Word Cloud per Topik")
            plot_topic_wordclouds(lda_model, st.session_state.lda_params['num_topics'])
            
            # Topic Distribution
            st.markdown("### 📊 Distribusi Topik")
            plot_topics_distribution(corpus, lda_model)
            
            # Dominant Topics
            st.markdown("### 🎯 Topik Dominan")
            plot_topic_dominance(corpus, lda_model)
            
            # Model Evaluation
            st.markdown("### 📈 Evaluasi Model")
            
            # Calculate coherence score
            coherence_model = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
            coherence_score = coherence_model.get_coherence()
            
            # Calculate perplexity
            perplexity = lda_model.log_perplexity(corpus)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coherence Score", f"{coherence_score:.4f}")
            with col2:
                st.metric("Perplexity", f"{perplexity:.4f}")
            
            # Interactive LDA Visualization
            st.markdown("### 🔬 Visualisasi Interaktif LDA")
            if st.button("Generate Interactive LDA Visualization"):
                with st.spinner("Creating interactive visualization..."):
                    create_pyldavis_visualization(lda_model, corpus, dictionary)
            
            # Store sample reviews for AI insights
            sample_reviews = df_filtered[text_col].head(5).tolist()
            st.session_state.sample_reviews = sample_reviews
            
            # AI-Powered Topic Insights
            if st.session_state.get("gemini_api_key"):
                st.markdown("### 🤖 AI-Powered Topic Insights")
                
                if st.button("Generate AI Insights", key="ai_insights_button"):
                    with st.spinner("Menganalisis dengan AI..."):
                        try:
                            # Get data from session state
                            topics_data = st.session_state.topics_data
                            sample_reviews = st.session_state.sample_reviews
                            
                            # Generate insights
                            ai_insights = generate_topic_insights(topics_data, sample_reviews)
                            
                            if ai_insights:
                                with st.container():
                                    st.markdown("""<div class='ai-response'>""", unsafe_allow_html=True)
                                    st.markdown("#### 🔍 Hasil Analisis AI")
                                    st.write(ai_insights)
                                    st.markdown("""</div>""", unsafe_allow_html=True)
                                    st.success("Analisis berhasil dihasilkan!")
                            else:
                                st.error("Gagal mendapatkan respons dari AI. Silakan coba lagi.")
                                
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat memproses: {str(e)}")
            else:
                st.warning(
                    "⚠️ Masukkan Gemini API Key di sidebar untuk mengaktifkan fitur AI.\n"
                    "Dapatkan API Key gratis di [Google AI Studio](https://aistudio.google.com/)"
                )            
            
            # Coherence Score Analysis
            st.markdown("### 📊 Analisis Coherence Score")
            if st.button("Analyze Optimal Number of Topics"):
                with st.spinner("Analyzing optimal number of topics..."):
                    model_list, coherence_values = evaluate_coherence_values(
                        processed_docs, dictionary, corpus, start=2, stop=12, step=1
                    )
                    
                    # Find optimal number of topics
                    optimal_topics = coherence_values.index(max(coherence_values)) + 2
                    st.success(f"Optimal number of topics: {optimal_topics}")
                    
                    # Display coherence values table
                    coherence_df = pd.DataFrame({
                        'Number of Topics': range(2, 12),
                        'Coherence Score': coherence_values
                    })
                    st.dataframe(coherence_df, use_container_width=True)
            
            # Download Results
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download Topics CSV"):
                    csv = topics_df.to_csv(index=False)
                    st.download_button(
                        label="Download Topics as CSV",
                        data=csv,
                        file_name="lda_topics.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Download Processed Data"):
                    processed_csv = df_filtered.to_csv(index=False)
                    st.download_button(
                        label="Download Processed Data as CSV",
                        data=processed_csv,
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
            
            # Save/Load Model
            st.markdown("### 💾 Model Management")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Model"):
                    # Save model to pickle
                    model_data = {
                        'lda_model': lda_model,
                        'dictionary': dictionary,
                        'corpus': corpus,
                        'topics_data': topics_data,
                        'coherence_score': coherence_score,
                        'perplexity': perplexity
                    }
                    
                    pickle_data = pickle.dumps(model_data)
                    st.download_button(
                        label="Download Model",
                        data=pickle_data,
                        file_name="lda_model.pkl",
                        mime="application/octet-stream"
                    )
            
            with col2:
                uploaded_model = st.file_uploader("Load Saved Model", type=['pkl'])
                if uploaded_model:
                    try:
                        model_data = pickle.load(uploaded_model)
                        st.session_state.lda_model = model_data['lda_model']
                        st.session_state.dictionary = model_data['dictionary']
                        st.session_state.corpus = model_data['corpus']
                        st.success("✅ Model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")

else:
    # Display instructions when no file is uploaded
    st.info("👆 Silakan upload file CSV atau Excel yang berisi data ulasan konsumen.")
    
    st.markdown("""
    ### 📋 Petunjuk Penggunaan:
    
    1. **Upload Data**: Upload file CSV/Excel yang berisi kolom ulasan konsumen
    2. **Konfigurasi**: Atur parameter preprocessing dan Gemini API key (opsional)
    3. **Preprocessing**: Sistem akan melakukan cleaning, tokenisasi, stemming, dan stopword removal
    4. **Analisis**: Lakukan analisis bigrams, asosiasi kata, dan topic modeling
    5. **Insights**: Dapatkan insight AI-powered untuk interpretasi hasil
    
    ### 📊 Fitur Utama:
    - **Topic Modeling** dengan LDA (Latent Dirichlet Allocation)
    - **Word Cloud** dan visualisasi bigrams
    - **Analisis asosiasi kata** dengan network visualization
    - **AI-powered insights** menggunakan Gemini
    - **Evaluasi model** dengan coherence score dan perplexity
    - **Interactive visualization** dengan pyLDAvis
    - **Export results** dalam format CSV dan pickle
    
    ### 🔧 Parameter yang Dapat Disesuaikan:
    - Jumlah topik (2-15)
    - Passes, chunk size, update frequency
    - Alpha dan eta parameters
    - Custom stopwords
    - Bigram settings (min count, threshold)
    
    ### 💡 Tips:
    - Gunakan file yang sudah berisi kolom 'ulasan' untuk hasil optimal
    - Masukkan Gemini API key untuk mendapatkan insights AI
    - Eksperimen dengan jumlah topik yang berbeda untuk hasil terbaik
    - Gunakan fitur coherence analysis untuk menentukan jumlah topik optimal
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>📊 Dashboard LDA - Topic Modeling untuk Review Konsumen Tokopedia</p>
        <p>Powered by Streamlit, Gensim, dan Gemeni</p>
    </div>
    """, unsafe_allow_html=True)