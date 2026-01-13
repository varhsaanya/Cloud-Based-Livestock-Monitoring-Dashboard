import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Livestock Monitoring System",
    page_icon='📍',
    layout="wide",
)
# Firebase configuration and initialization

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            'credentials-to-your-project-private-key.json'})
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = initialize_firebase()


# Fetch data from Firestore
def fetch_data():
    docs = db.collection('tracking').get()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)

    
    
    if 'tags' in df.columns:
        df['tags'] = df['tags'].astype(str)
    
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    return df


df = fetch_data()



# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Sidebar

def calculate_orientation(x, y, z):
    x_norm = x / 20  
    y_norm = y / 20  
    z_norm = z / 110  
    
    # angle of tilt from vertical
    tilt = np.arccos(z_norm / np.sqrt(x_norm*2 + y_norm*2 + z_norm*2))
    # angle in the x-y plane
    rotation = np.arctan2(y_norm, x_norm)
    return tilt, rotation

def determine_position(tilt):
    return 'lying' if tilt > np.pi/4 else 'standing'


with st.sidebar:
    st.title('Animal Tracking System')
    
    if 'tags' in df.columns:
        tag_list = df['tags'].unique()
        tag_list.sort()
        
        selected_single = st.selectbox('Select a cow for trajectory', ['None'] + list(tag_list))
        
        selected_multi = st.multiselect('Select multiple cows for current locations', list(tag_list))
    else:
        st.error("No 'tags' column found in the data.")

# Main content area
if all(col in df.columns for col in ['tags', 'latitude', 'longitude', 'gyro_x', 'gyro_y', 'gyro_z']):
    if selected_single != 'None':
        df_single = df[df['tags'] == selected_single].reset_index(drop=True)
    else:
        df_single = pd.DataFrame()

    df_multi = df[df['tags'].isin(selected_multi)]
    
    df_map = pd.concat([df_single, df_multi]).dropna(subset=['latitude', 'longitude'])
    
    if not df_map.empty:
        latest_points = df_multi.groupby('tags').last().reset_index()
        fig = go.Figure()
        
        # Plot trajectory for single selected cow
        if not df_single.empty:
            df_single['tilt'], df_single['rotation'] = zip(*df_single.apply(lambda row: calculate_orientation(row['gyro_x'], row['gyro_y'], row['gyro_z']), axis=1))
            df_single['position'] = df_single['tilt'].apply(determine_position)
    
    
            st.write("Sample of calculated data:")
            st.write(df_single[['latitude', 'longitude', 'tilt', 'rotation', 'position']].head())
    
    
            line_length = 0.0005  
            df_single['end_lat'] = df_single['latitude'] + line_length * np.cos(df_single['rotation'])
            df_single['end_lon'] = df_single['longitude'] + line_length * np.sin(df_single['rotation'])
    
    
            for index, row in df_single.iterrows():
                fig.add_trace(go.Scattermapbox(
                    lat=[row['latitude'], row['end_lat']],
                    lon=[row['longitude'], row['end_lon']],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    showlegend=False
            ))
    
            fig.add_trace(go.Scattermapbox(
                lat=df_single['latitude'],
                lon=df_single['longitude'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df_single['position'].map({'standing': 'green', 'lying': 'red'}),
                    symbol='circle',
                ),
                text=[f"<b>Position: {pos}</b><br>Point: {i}<br>Tilt: {tilt:.2f}<br>Rotation: {rot:.2f}<br>Gyro: ({x}, {y}, {z})" 
                    for i, (pos, tilt, rot, x, y, z) in enumerate(zip(df_single['position'], df_single['tilt'], 
                                                                df_single['rotation'], df_single['gyro_x'], 
                                                                df_single['gyro_y'], df_single['gyro_z']))],
                hoverinfo='text',
                name='Cow Position'
            ))
    
            fig.add_trace(go.Scattermapbox(
                lat=df_single['latitude'],
                lon=df_single['longitude'],
                mode='lines',
                line=dict(width=2, color='gray'),
                name='Trajectory'
            ))

            fig.add_trace(go.Scattermapbox(
                lat=[df_single['latitude'].iloc[0]],
                lon=[df_single['longitude'].iloc[0]],
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Standing',
                showlegend=True,
                visible=True
            ))
            fig.add_trace(go.Scattermapbox(
                lat=[df_single['latitude'].iloc[0]],
                lon=[df_single['longitude'].iloc[0]],
             mode='markers',
            marker=dict(size=10, color='red'),
            name='Lying',
        showlegend=True,
        visible=True
    ))

        if not df_multi.empty:
            fig.add_trace(go.Scattermapbox(
                lat=latest_points['latitude'],
                lon=latest_points['longitude'],
                mode='markers',
                marker=dict(size=15, color='blue'),
                text=latest_points['tags'],
                hoverinfo='text',
                name='Latest Positions'
            ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(
                    lat=df_map['latitude'].mean(),
                    lon=df_map['longitude'].mean()
                ),
                zoom=10
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=600,
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("No valid latitude and longitude data for the selected cows.")
else:
    st.error("Required 'tags', 'latitude', 'longitude', 'gyro_x', 'gyro_y', and 'gyro_z' columns not found in the data.")
