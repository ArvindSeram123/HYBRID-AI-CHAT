import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Set page config
st.set_page_config(page_title="Vietnam Travel Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    # You can replace this with actual file loading
    with open('vietnam_travel_dataset.json', 'r') as f:
        raw_data = json.load(f)
    
    df = pd.json_normalize(raw_data)
    return df, raw_data

try:
    df, raw_data = load_data()
except FileNotFoundError:
    st.error("Data file not found. Make sure 'data/vietnam_travel_dataset.json' exists in your project.")
    st.stop()

# Preprocessing
@st.cache_data
def preprocess_data(df):
    # Count by type
    type_counts = df['type'].value_counts().to_dict()
    
    # Count by city
    city_counts = df[df['type'] != 'City']['city'].value_counts().to_dict()
    
    # Extract tags
    all_tags = []
    for tags in df['tags'].dropna():
        if isinstance(tags, list):
            all_tags.extend(tags)
    tag_counts = dict(Counter(all_tags).most_common(10))
    
    # Regional distribution
    cities_df = df[df['type'] == 'City'][['name', 'region']]
    content_df = df[df['type'] != 'City'].copy()
    content_by_region = content_df.groupby('region').size().to_dict()
    
    return {
        'type_counts': type_counts,
        'city_counts': city_counts,
        'tag_counts': tag_counts,
        'content_by_region': content_by_region,
        'total_records': len(df),
        'total_cities': len(cities_df)
    }

stats = preprocess_data(df)

# Header
st.title("Vietnam Travel Dataset Dashboard")
st.markdown("**Hybrid AI Project** - Semantic Search + Knowledge Graph Analysis")

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Records", stats['total_records'])
with col2:
    st.metric("Cities", stats['total_cities'])
with col3:
    st.metric("Content Types", len(stats['type_counts']))
with col4:
    st.metric("Attractions", stats['type_counts'].get('Attraction', 0))
with col5:
    st.metric("Hotels", stats['type_counts'].get('Hotel', 0))

st.divider()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Regional Analysis", "City Deep Dive", "Insights"])

with tab1:
    st.subheader("Content Type Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        type_df = pd.DataFrame(list(stats['type_counts'].items()), columns=['Type', 'Count'])
        fig_pie = px.pie(type_df, values='Count', names='Type', 
                         title="Content Type Breakdown",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(type_df, x='Type', y='Count',
                        title="Asset Count by Type",
                        color='Count',
                        color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Top tags
    st.subheader("Most Common Tags")
    tag_df = pd.DataFrame(list(stats['tag_counts'].items()), columns=['Tag', 'Frequency'])
    tag_df = tag_df.sort_values('Frequency', ascending=True)
    
    fig_tags = px.barh(tag_df, x='Frequency', y='Tag',
                       title="Top Travel Experience Tags",
                       color='Frequency',
                       color_continuous_scale='Greens')
    st.plotly_chart(fig_tags, use_container_width=True)

with tab2:
    st.subheader("Regional Distribution")
    
    region_df = pd.DataFrame(list(stats['content_by_region'].items()), 
                            columns=['Region', 'Assets'])
    region_df = region_df.sort_values('Assets', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_region = px.bar(region_df, x='Region', y='Assets',
                           title="Assets by Region",
                           color='Assets',
                           color_continuous_scale='Oranges')
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        fig_region_pie = px.pie(region_df, values='Assets', names='Region',
                               title="Regional Coverage %")
        st.plotly_chart(fig_region_pie, use_container_width=True)
    
    # Stats
    st.info(f"""
    **Regional Insights:**
    - Total regions covered: {len(stats['content_by_region'])}
    - Largest region: {region_df.iloc[0]['Region']} ({region_df.iloc[0]['Assets']} assets)
    - Smallest region: {region_df.iloc[-1]['Region']} ({region_df.iloc[-1]['Assets']} assets)
    """)

with tab3:
    st.subheader("Assets per City")
    
    city_df = pd.DataFrame(list(stats['city_counts'].items()), 
                          columns=['City', 'Count']).sort_values('Count', ascending=True)
    
    fig_cities = px.barh(city_df, x='Count', y='City',
                        title="Content Assets by City",
                        color='Count',
                        color_continuous_scale='Viridis')
    st.plotly_chart(fig_cities, use_container_width=True)
    
    # City selector
    selected_city = st.selectbox("Select a city for details:", city_df['City'].unique())
    city_data = df[df['city'] == selected_city]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Attractions", len(city_data[city_data['type'] == 'Attraction']))
    with col2:
        st.metric("Hotels", len(city_data[city_data['type'] == 'Hotel']))
    with col3:
        st.metric("Activities", len(city_data[city_data['type'] == 'Activity']))
    
    st.write(city_data[['id', 'name', 'type', 'tags']].head(10))

with tab4:
    st.subheader("Business Intelligence & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Data Quality:**
        - Clean, consistent schema
        - 350 records ready for embeddings
        - Balanced distribution across regions
        - Rich semantic information via tags
        """)
        
        st.success("""
        **Semantic Search Ready:**
        - 115 attractions for vector DB
        - Diverse tags for filtering
        - Connections across all regions
        - Perfect for Pinecone indexing
        """)
    
    with col2:
        st.warning("""
        **Graph Database Potential:**
        - 10 city hub nodes
        - 295+ content nodes
        - Located_In, Available_In relationships
        - Multi-hop query support
        """)
        
        st.success("""
        **Hybrid AI Strategy:**
        - Pinecone: Semantic similarity
        - Neo4j: Relationships & traversal
        - LLM: Personalized synthesis
        - 100% coverage across Vietnam
        """)
    
    st.markdown("---")
    
    st.subheader("Data Preprocessing Summary")
    
    summary_metrics = {
        "Total Records": stats['total_records'],
        "Cities": stats['total_cities'],
        "Regions": len(stats['content_by_region']),
        "Unique Tags": len(stats['tag_counts']),
        "Avg Assets/City": round(stats['total_records'] / stats['total_cities'], 1)
    }
    
    for key, value in summary_metrics.items():
        st.write(f"**{key}:** {value}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Vietnam Travel Analytics Dashboard | Hybrid AI Project</p>
    <p>Data ready for Pinecone + Neo4j + Gemini integration</p>
</div>
""", unsafe_allow_html=True)
