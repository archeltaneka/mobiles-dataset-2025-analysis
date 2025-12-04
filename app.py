from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.analytics.scoring import calculate_value_score
from src.analytics.clustering import assign_phone_clusters
from src.cleaning.pipeline import clean_data
from src.data.mobiles import MobilesData


# Page configuration
st.set_page_config(
    page_title="Smartphone Market Dashboard 2025",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px !important;
        font-weight: bold;
        color: #D3C3C3;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 20px !important;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    """

    mobiles_data = MobilesData(data_dir='data', fname='Mobiles Dataset (2025).csv')
    df = mobiles_data.load_data()
    df = clean_data(df)
    df = calculate_value_score(df)
    df = assign_phone_clusters(df)

    return df


# Load data
df = load_data()

# Sidebar
# st.sidebar.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Smartphone+Dashboard", use_container_width=True)
st.sidebar.title("🎛️ Filters")

# Currency selection
currency_map = {
    'USD ($)': 'launched_price_usa',
    'INR (₹)': 'launched_price_india',
    'AED (د.إ)': 'launched_price_dubai',
    'PKR (₨)': 'launched_price_pakistan',
    'CNY (¥)': 'launched_price_china'
}
selected_currency = st.sidebar.selectbox('Currency', list(currency_map.keys()))
price_col = currency_map[selected_currency]

# Manufacturer filter
manufacturers = ['All'] + sorted(df['company_name'].unique().tolist())
selected_manufacturer = st.sidebar.selectbox('company_name', manufacturers)

# Budget tier filter
budget_tiers = ['All', 'Budget', 'Mid', 'Flagship']
selected_tier = st.sidebar.selectbox('Budget Tier', budget_tiers)

# Year filter
years = ['All'] + sorted(df['launched_year'].unique().tolist(), reverse=True)
selected_year = st.sidebar.multiselect('Launch Year', years, default=['All'])

# Price range filter
min_price, max_price = int(df[price_col].min()), int(df[price_col].max())
price_range = st.sidebar.slider(
    'Price Range',
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

# Apply filters
filtered_df = df.copy()

if selected_manufacturer != 'All':
    filtered_df = filtered_df[filtered_df['company_name'] == selected_manufacturer]

if selected_tier != 'All':
    filtered_df = filtered_df[filtered_df['phone_cluster'] == selected_tier]

if 'All' not in selected_year:
    filtered_df = filtered_df[filtered_df['launched_year'].isin(selected_year)]

filtered_df = filtered_df[
    (filtered_df[price_col] >= price_range[0]) & 
    (filtered_df[price_col] <= price_range[1])
]

# Header
st.markdown('<p class="main-header">📱 Smartphone Market Dashboard 2025</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header"> \
    This interactive dashboard provides a comprehensive exploration of the 2025 smartphone market.\
    </p>', 
    unsafe_allow_html=True)
st.markdown(
    f'<p> \
    Users can analyze device specifications, compare prices across segments, and discover insights \
    into how different features influence a phone\'s value. The dataset is sourced from \
    <a href="https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025">Kaggle</a> \
    and includes detailed specs such as RAM, storage, cameras, screen size, battery capacity, \
    weight, and launch price.\
    Key features include price distribution analysis, value scoring, correlation insights, \
    market segmentation (Budget, Mid-range, Flagship), and interactive comparisons across brands \
    and device specifications.\n\n This dashboard aims to assist analysts, enthusiasts, and buyers \
    in understanding market trends through clear and intuitive visualizations.\
    </p>',
    unsafe_allow_html=True
)

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Models", len(filtered_df), f"{len(filtered_df)} phones")
    
with col2:
    avg_price = filtered_df[price_col].mean()
    currency_symbol = selected_currency.split('(')[1].split(')')[0]
    st.metric("Avg Price", f"{currency_symbol}{avg_price:,.0f}")
    
with col3:
    avg_battery = filtered_df['battery_capacity'].mean()
    st.metric("Avg Battery", f"{avg_battery:,.0f} mAh")
    
with col4:
    avg_ram = filtered_df['ram'].mean()
    st.metric("Avg RAM", f"{avg_ram:.1f} GB")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "⚙️ Specifications", "💎 Value Finder", "📈 Price Analysis"])

with tab1:
    st.header("Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top manufacturers
        st.subheader("Top 10 Manufacturers")
        manufacturer_counts = filtered_df['company_name'].value_counts().head(10).sort_values(ascending=True)
        fig = px.bar(
            x=manufacturer_counts.values,
            y=manufacturer_counts.index,
            orientation='h',
            labels={'x': 'Number of Models', 'y': 'Manufacturer'},
            text=manufacturer_counts.values  # Add text labels
        )
        fig.update_traces(
            marker_color='#3b82f6',  # Solid color instead of gradient
            textposition='outside',   # Position text outside the bars
            textfont=dict(size=10)
        )
        fig.update_layout(
            height=400, 
            showlegend=False,
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(200, 200, 200, 0.3)',  # Light gray with transparency
                griddash='dash'  # Dashed grid lines
            ),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Phones launched over time
        st.subheader("Phones Launched Over Time")
        launch_trend = filtered_df.groupby('launched_year').size().reset_index(name='Count')
        fig = px.line(
            launch_trend,
            x='launched_year',
            y='Count',
            markers=True,
            labels={'Count': 'Number of Phones'}
        )
        fig.update_traces(line_color='#8b5cf6', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(f"Price Distribution ({selected_currency.split('(')[0].strip()})")

        # Histogram
        hist = go.Histogram(
            x=filtered_df[price_col],
            nbinsx=50,
            name="Histogram",
            marker_color="#10b981",
            opacity=0.6
        )

        # KDE curve
        kde = gaussian_kde(filtered_df[price_col])
        x_vals = np.linspace(filtered_df[price_col].min(), filtered_df[price_col].max(), 200)
        y_vals = kde(x_vals)

        # Scale KDE to histogram
        bin_width = (filtered_df[price_col].max() - filtered_df[price_col].min()) / 50
        y_vals_scaled = y_vals * len(filtered_df[price_col]) * bin_width

        kde_line = go.Scatter(
            x=x_vals,
            y=y_vals_scaled,
            mode="lines",
            name="KDE",
            line=dict(color="#CF5699", width=2)
        )

        # Create figure
        fig = go.Figure(data=[hist, kde_line])

        # Vertical dashed line for mean price
        mean_price = filtered_df[price_col].mean()
        fig.add_shape(
            type="line",
            x0=mean_price, x1=mean_price,
            y0=0, y1=y_vals_scaled.max(),
            line=dict(color="#E44C5A", width=2, dash="longdash")
        )

        # Annotation
        fig.add_annotation(
            x=mean_price,
            y=y_vals_scaled.max(),
            text=f"Mean price = {selected_currency.split('(')[0].strip()} {mean_price:,.0f}",
            showarrow=False,
            yshift=10,
            font=dict(color="#E44C5A", size=16, weight='bold')
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Top processors
        st.subheader("Top Processor Types")
        processor_counts = filtered_df['processor_type'].value_counts().head(6)
        fig = px.pie(
            values=processor_counts.values,
            names=processor_counts.index,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Processor manufacturer distribution
    st.subheader("Processor Manufacturer Distribution")
    proc_manuf_counts = filtered_df['processor_name'].value_counts()
    fig = px.bar(
        x=proc_manuf_counts.index,
        y=proc_manuf_counts.values,
        labels={'x': 'Processor Manufacturer', 'y': 'Count'},
        text=proc_manuf_counts.values
    )
    fig.update_traces(
        marker_color='#3b82f6',  # Solid color instead of gradient
        textposition='outside',   # Position text outside the bars
        textfont=dict(size=10)
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Specifications Analysis")
    
    # Specification selector
    spec_options = {
        'RAM (GB)': 'ram',
        'Storage (GB)': 'internal_memory',
        'Battery (mAh)': 'battery_capacity',
        'Screen Size (inches)': 'screen_size',
        'Front Camera (MP)': 'front_camera',
        'Back Camera (MP)': 'back_camera',
        'Weight (g)': 'mobile_weight'
    }
    
    selected_spec = st.selectbox('Select Specification', list(spec_options.keys()))
    spec_col = spec_options[selected_spec]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution of selected spec
        st.subheader(f"{selected_spec} Distribution")
        fig = px.histogram(
            filtered_df,
            x=spec_col,
            nbins=15,
            labels={spec_col: selected_spec},
            color_discrete_sequence=['#ec4899']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price vs selected spec
        st.subheader(f"Price vs {selected_spec}")
        fig = px.density_heatmap(
            filtered_df,
            x=spec_col,
            y=price_col,
            nbinsx=30,
            nbinsy=30,
            labels={spec_col: selected_spec, price_col: "Price"},
            color_continuous_scale="Blackbody"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Multiple specifications distributions
    st.subheader("All Specifications Overview")
    
    specs_to_plot = ['ram', 'internal_memory', 'battery_capacity', 'back_camera']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['RAM (GB) Distribution', 'Storage (GB) Distribution', 
                       'Battery (mAH) Distribution', 'Camera (MP) Distribution']
    )
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899']
    
    for idx, (spec, color) in enumerate(zip(specs_to_plot, colors)):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        hist_data = filtered_df[spec].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hist_data.index, y=hist_data.values, marker_color=color, showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("💎 Value-for-Money Analysis")
    
    st.markdown("""
        These phones offer the best specifications relative to their price.  
        The **Value Score** represents how much performance a device delivers for each dollar spent.

        The score is built from a weighted combination of key specifications, giving higher importance to 
        **RAM** and **internal storage**, while assigning smaller but meaningful weights to screen size, 
        battery capacity, phone weight, and camera quality.

        More formally:

        $$
        \\text{Spec Score} =
        0.30 \\cdot \\text{RAM} +
        0.30 \\cdot \\text{Internal Memory} +
        0.08 \\cdot \\text{Screen Size} +
        0.08 \\cdot \\text{Battery} +
        0.08 \\cdot \\text{Weight} +
        0.08 \\cdot \\text{Front Camera} +
        0.08 \\cdot \\text{Back Camera}
        $$

        Finally, the **Value Score** normalizes this performance score by price:

        $$
        \\text{Value Score} = 
        \\frac{\\text{Spec Score}}{\\text{Launched Price}}.
        $$
        """
    )
    
    # Budget tier selector for value analysis
    value_tier = st.radio(
        "Select Budget Tier",
        ['All', 'Budget (<$250)', 'Mid-Range (\$250-\$600)', 'Flagship ($600+)'],
        horizontal=True
    )
    
    # Filter based on tier
    value_df = filtered_df.copy()
    if value_tier == 'Budget (<$250)':
        value_df = value_df[value_df['phone_cluster'] == 'Budget']
    elif value_tier == 'Mid-Range (\$250-\$600)':
        value_df = value_df[value_df['phone_cluster'] == 'Mid']
    elif value_tier == 'Flagship ($600+)':
        value_df = value_df[value_df['phone_cluster'] == 'Flagship']
    
    # Top 10 value phones
    top_value = value_df.nlargest(10, 'Value_Score')[
        ['model_series', price_col, 'ram', 'internal_memory', 
         'battery_capacity', 'back_camera', 'Value_Score']
    ].reset_index(drop=True)
    
    top_value.index = top_value.index + 1
    
    # Display as styled dataframe
    st.subheader(f"🏆 Top 10 Value Phones - {value_tier}")
    
    # Format the dataframe
    styled_df = top_value.copy()
    styled_df.columns = ['model_series', price_col, 'ram', 'internal_memory', 
                         'battery_capacity', 'back_camera', 'Value_Score']
    
    st.dataframe(
        styled_df.style.background_gradient(subset=['Value_Score'], cmap='RdYlGn')
                       .format({
                           'Price': '{:,.0f}',
                           'Value Score': '{:.2f}'
                       }),
        use_container_width=True,
        height=400
    )
    
    # Value score visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Value Score Comparison")
        fig = px.bar(
            top_value,
            x='Value_Score',
            y='model_series',
            orientation='h',
            color='Value_Score',
            color_continuous_scale='RdYlGn',
            labels={'Value_Score': 'Value Score', 'model_series': ''}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Value Score")
        fig = px.scatter(
            top_value,
            x=price_col,
            y='Value_Score',
            size='ram',
            color='model_series',
            hover_data={'ram': True, 'battery_capacity': True, price_col: ':,.0f'},
            labels={price_col: 'Price', 'Value_Score': 'Value Score'},
            size_max=25
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Price Trends & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average price over time
        st.subheader(f"Average Price Trend ({selected_currency.split('(')[0].strip()})")
        avg_price_trend = filtered_df.groupby('launched_year')[price_col].mean().reset_index()
        fig = px.line(
            avg_price_trend,
            x='launched_year',
            y=price_col,
            markers=True,
            labels={price_col: 'Average Price'}
        )
        fig.update_traces(line_color='#10b981', line_width=3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price by manufacturer
        st.subheader("Average Price by Manufacturer")
        avg_price_manuf = filtered_df.groupby('company_name')[price_col].mean().sort_values(ascending=True).head(10)
        fig = px.bar(
            x=avg_price_manuf.values,
            y=avg_price_manuf.index,
            orientation='h',
            labels={'x': f'Average Price ({selected_currency.split("(")[0].strip()})', 'y': 'Manufacturer'},
            text=[f'{price:.1f}' for price in avg_price_manuf.values],
        )
        fig.update_layout(height=400, showlegend=False)
        fig.update_traces(
            marker_color='#3b82f6',  # Solid color instead of gradient
            textposition='outside',   # Position text outside the bars
            textfont=dict(size=10)
        )
        fig.update_layout(
            height=400, 
            showlegend=False,
            xaxis=dict(
                showgrid=True, 
                gridcolor='rgba(200, 200, 200, 0.3)',  # Light gray with transparency
                griddash='dash'  # Dashed grid lines
            ),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Specification trends over time
    st.subheader("Specification Evolution Over Time")
    
    spec_evolution = filtered_df.groupby('launched_year').agg({
        'ram': 'mean',
        'internal_memory': 'mean',
        'battery_capacity': 'mean',
        'back_camera': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Average RAM Over Time', 'Average Storage Over Time',
                       'Average Battery Over Time', 'Average Camera Over Time']
    )
    
    specs = [
        ('ram', 'RAM (GB)', '#3b82f6'),
        ('internal_memory', 'Storage (GB)', '#10b981'),
        ('battery_capacity', 'Battery (mAh)', '#f59e0b'),
        ('back_camera', 'Camera (MP)', '#ec4899')
    ]
    
    for idx, (spec, label, color) in enumerate(specs):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        fig.add_trace(
            go.Scatter(
                x=spec_evolution['launched_year'],
                y=spec_evolution[spec],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=8),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation polar bar visualization
    st.subheader("Phone Specifications Effect on Price")

    corr_cols = ['ram', 'internal_memory', 'battery_capacity', 'screen_size', 
                'back_camera', 'mobile_weight', price_col]

    # Compute correlation with price
    corr_series = filtered_df[corr_cols].corr()[price_col].iloc[:-1]

    corr_df = corr_series.reset_index()
    corr_df.columns = ['specification', 'price_correlation']
    corr_df['specification'] = ['RAM', 'Internal Memory', 'Battery Capacity', 'Screen Size', 'Back Camera', 'Mobile Weight']

    # Sort by correlation strength
    corr_df = corr_df.sort_values('price_correlation', ascending=False)

    # Convert to percentage for better readability
    corr_df['correlation_pct'] = (corr_df['price_correlation'] * 100).round(1)

    # Create enhanced polar bar plot (no `text=` here)
    fig = px.bar_polar(
        corr_df,
        r="price_correlation",
        theta="specification",
        color="price_correlation",
        color_continuous_scale=["#ef4444", "#fbbf24", "#10b981"],  # Red-Yellow-Green
        hover_data={'price_correlation': ':.3f', 'correlation_pct': True}
    )

    # Add in-text labels and style traces
    fig.update_traces(
        text=corr_df['correlation_pct'].astype(str) + '%',
        marker=dict(line=dict(color='white', width=2)),
        hovertemplate=(
            "<b>%{theta}</b><br>"
            "Correlation: %{r:.3f}<br>"
            "Percent: %{text}<extra></extra>"
        )
    )

    # Configure hover template for clarity
    fig.update_traces(
        hovertemplate='<b>%{theta}</b><br>Correlation: %{customdata[0]:.3f}<br>Percent: %{text}<extra></extra>',
        # attach customdata for the hovertemplate (plotly will pick from hover_data if present)
    )

    # Make sure radial range isn't zero
    max_r = max(corr_df['price_correlation'].max(), 0.01)
    fig.update_layout(
        height=550,
        polar=dict(
            bgcolor="rgba(17, 24, 39, 0.95)",
            radialaxis=dict(
                showticklabels=True,
                tickfont=dict(size=10, color='white'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                range=[0, max_r * 1.15]
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='white', family='Arial'),
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.3)'
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <p>📱 Smartphone Market Dashboard 2025</p>
    <p>Built with Streamlit & Plotly • Data from Kaggle</p>
    <p>© 2025 <strong>Archel Taneka Sutanto</strong>. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)