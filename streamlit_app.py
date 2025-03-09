# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import leafmap
import folium
from streamlit_folium import st_folium, folium_static
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import pydeck as pdk

from sklearn.preprocessing import MinMaxScaler

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -10px;
        margin-bottom: -10px;
        margin-left: -10px;
        margin-right: -10px;
    }
    img[data-testid="stLogo"] {
                height: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Historical U.S. Presidential Election")
st.divider()

@st.cache_data
def load_data():
    # Results by State
    election = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_presidential_election_results_by_state', header=0)[0].drop(columns=['Unnamed: 5', 'Unnamed: 16', 'Unnamed: 27', 'Unnamed: 38', 'Unnamed: 49', 'Unnamed: 60', 'State.1'])
    election.columns = election.columns.str.replace('[^0-9a-zA-Z_]', '', regex=True).str.strip().str.replace(' ', '_')
    election = election[election.State.notna()].copy()
    election = election.melt(id_vars='State', value_name='Result', var_name='Year')
    election = election[election.State != 'State'].copy()

    # National results
    pres = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_presidential_elections_by_popular_vote_margin')[2]
    pres.columns = ['_'.join(col).strip() for col in pres.columns.values]
    pres.columns = pres.columns.str.replace('[^0-9a-zA-Z_]', '', regex=True).str.strip().str.replace(' ', '_')
    pres = pres[['Election_Election', 'Winnerandparty_Winnerandparty', 'Winnerandparty_Winnerandparty1']].rename(columns={'Election_Election' : 'Year', 'Winnerandparty_Winnerandparty' : 'President', 'Winnerandparty_Winnerandparty1' : 'Result'})
    pres['Year'] = np.where(pres.Year == '1788–89', '1789', pres.Year)
    pres['Result'] = pres.Result.map({'Ind.' : 'I', 'D-R' : 'DR', 'Fed.' : 'F', 'Dem.' : 'D', 'Rep.' : 'R', 'Whig' : 'W'})
    pres['State'] = ' National'

    # Merge
    election = pd.concat([election, pres.drop(columns=['President'])])
    election = pd.merge(election, pres.drop(columns=['Result', 'State']), on='Year', how='left')

    # Party full name
    election['Result_FullName'] = election.Result.map({"R": "Republican", "D": "Democratic", "DR": "Democratic-Republican", "W": "Whig", "F": "Federalist", "GW": "George Washington", "NR": "National Republican", "SD": "Southern Democrat", "BM": "Progressive \"Bull Moose\"", "LR": "Liberal Republican", "AI": "American Independent", "SR": "States' Rights", "PO": "Populist", "CU": "Constitutional Union", "I": "Independent", "PR": "Progressive", "ND": "Northern Democrat", "KN": "Know Nothing", "AM": "Anti-Masonic", "N": "Nullifier", "SP": "Split evenly"})

    # Party as categorical variable
    election['Result_Norm'] = pd.Categorical(election['Result']).codes

    # Plus or minus
    election['Result_plusminus'] = np.select(
        [
            election['Result'].isin(['D', 'DR', 'SD', 'BM', 'LR', 'PR', 'ND', 'PO', 'Jackson']),
            election['Result'].isin(['R', 'W', 'F', 'NR', 'AI', 'SR', 'CU', 'KN', 'AM', 'N', 'Adams', 'Crawford', 'Clay']),
            election['Result'].isin(['GW', 'I', 'SP'])
        ],
        [-1, 1, 0],
        default=0
    )

    # Hover Text for heatmap
    election = election.sort_values(['State', 'Year'])
    election['hover_text1'] = (
        '<b>State: ' + election.State + '</b><br>' +
        'Year: ' + election.Year.astype(str) + '<br>' +
        'Result: ' + election.Result_FullName.astype(str) + '<br>'
    )

    return election
election_full = load_data()


with st.sidebar:
    st.logo(path_cda + '\\0_Branding\\logo_large.png', size='large')
    st.subheader('Election Year')
    select_year = st.slider('', min_value=int(election_full.Year.min()), max_value=int(election_full.Year.max()), value=(int(election_full.Year.min()), int(election_full.Year.max())))

#
#

election = election_full[(election_full.Year.astype(int) >= select_year[0]) & (election_full.Year.astype(int) <= select_year[1])]
election['Result_Cumul'] = election.groupby('State').Result_plusminus.cumsum()

poly_states = gpd.read_file(path_data + '\\cb_2018_us_state_20m.shp')
poly_states['NAME'] = np.where(poly_states.NAME == 'District of Columbia', 'D.C.', poly_states.NAME)
poly_states = poly_states[poly_states.GEOID != '72'].copy()
election_gdf = pd.merge(poly_states,
                        election[election.State != ' National'],
                        left_on='NAME',
                        right_on='State',
                        how='left')
election_gdf['date'] = election_gdf.apply(lambda x: dt.datetime(int(x.Year), 11,5), axis=1)


def page0():
    st.header('Extract Data')
    st.code('''
    import pandas as pd
            
    # Results by State
    election = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_presidential_election_results_by_state', header=0)[0].drop(columns=['Unnamed: 5', 'Unnamed: 16', 'Unnamed: 27', 'Unnamed: 38', 'Unnamed: 49', 'Unnamed: 60', 'State.1'])
    election.columns = election.columns.str.replace('[^0-9a-zA-Z_]', '', regex=True).str.strip().str.replace(' ', '_')
    election = election[election.State.notna()].copy()
    election = election.melt(id_vars='State', value_name='Result', var_name='Year')
    election = election[election.State != 'State'].copy()

    # National results
    pres = pd.read_html('https://en.wikipedia.org/wiki/List_of_United_States_presidential_elections_by_popular_vote_margin')[2]
    pres.columns = ['_'.join(col).strip() for col in pres.columns.values]
    pres.columns = pres.columns.str.replace('[^0-9a-zA-Z_]', '', regex=True).str.strip().str.replace(' ', '_')
    pres = pres[['Election_Election', 'Winnerandparty_Winnerandparty', 'Winnerandparty_Winnerandparty1']].rename(columns={'Election_Election' : 'Year', 'Winnerandparty_Winnerandparty' : 'President', 'Winnerandparty_Winnerandparty1' : 'Result'})
    pres['Year'] = np.where(pres.Year == '1788–89', '1789', pres.Year)
    pres['Result'] = pres.Result.map({'Ind.' : 'I', 'D-R' : 'DR', 'Fed.' : 'F', 'Dem.' : 'D', 'Rep.' : 'R', 'Whig' : 'W'})
    pres['State'] = ' National'

    # Merge
    election = pd.concat([election, pres.drop(columns=['President'])])
    election = pd.merge(election, pres.drop(columns=['Result', 'State']), on='Year', how='left')
    ''')
    st.header('Output')
    st.dataframe(election.drop(columns=['Result_Norm', 'Result_plusminus', 'hover_text1', 'Result_Cumul']))



# Heatmap of actual parties
#-------------------------------------------------------------
def page1():
    st.header('Wins by State')

    palette = px.colors.qualitative.Alphabet[:25]
    palette[0] = 'white'
    palette[8] = "blue"
    palette[20] = "red"
    palette = [[item1, item2] for item1, item2 in zip(election.groupby(['Result_Norm']).size().reset_index().sort_values('Result_Norm').Result_Norm.tolist(), palette)]
    palette = [[(color[0] - palette[0][0]) / (palette[-1][0] - palette[0][0]), color[1]] for color in palette]

    fig = go.Figure(data=go.Heatmap(
        z=election.Result_Norm,
        x=election.Year,
        y=election.State,
        hoverinfo='text',
        text=election.hover_text1,
        colorscale=palette,
        hovertemplate='%{text}',
        showscale=False
    ))
    fig.update_yaxes(
        tickvals=election.State,
        ticktext=[f"<span style='color: gold; font-size: 22px;'>{state}</span>" if state == ' National' else state for state in election.State],
    )
    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        height=1000
    )
    st.plotly_chart(fig)

    st.code('''
        import pandas as pd
              
        # Party as numeric variable
        election['Result_Norm'] = pd.Categorical(election['Result']).codes

        # Hover Text for heatmap
        election = election.sort_values(['State', 'Year'])
        election['hover_text1'] = (
            '<b>State: ' + election.State + '</b><br>' +
            'Year: ' + election.Year.astype(str) + '<br>' +
            'Result: ' + election.Result_FullName.astype(str) + '<br>'
        )
    ''')
    st.dataframe(election.drop(columns=['Result_plusminus', 'Result_Cumul']))
    st.code(''' 
        import plotly.graph_objects as go
        import streamlit as st

        # Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=election.Result_Norm,
            x=election.Year,
            y=election.State,
            hoverinfo='text',
            text=election.hover_text1,
            colorscale=palette,
            hovertemplate='%{text}',
            showscale=False
        ))
        fig.update_yaxes(
            tickvals=election.State,
            ticktext=[f"<span style='color: gold; font-size: 22px;'>{state}</span>" if state == ' National' else state for state in election.State],
        )
        fig.update_layout(
            yaxis=dict(autorange='reversed'),
            height=1000
        )
            
        # Display in Streamlit
        st.plotly_chart(fig)
    ''')




# Heatmap of R vs D
#-------------------------------------------------------------
def page2():
    st.markdown("<h2 style='text-align: left;'>Cumulative <span style='color: blue;'>Left</span> vs. <span style='color: red;'>Right</span> Wins by State</h1>", unsafe_allow_html=True)

    fig = go.Figure(data=go.Heatmap(
        z=election.Result_Cumul,
        x=election.Year,
        y=election.State,
        colorscale=[[0, 'rgb(0, 66, 202)'], 
                    [0.25, 'rgb(102, 102, 255)'], 
                    [0.5, 'rgb(230, 230, 250)'], 
                    [0.75, 'rgb(255, 102, 102)'], 
                    [1, 'rgb(216, 24, 35)']],
        hoverinfo='text',
        text=election.hover_text1, 
        hovertemplate='%{text}',
        showscale=False
    ))
    fig.update_yaxes(
        tickvals=election.State,
        ticktext=[f"<span style='color: gold; font-size: 22px;'>{state}</span>" if state == ' National' else state for state in election.State],
    )
    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        height=1000
    )
    st.plotly_chart(fig)

    st.code('''
    # Plus or minus
    election['Result_plusminus'] = np.select(
        [
            election['Result'].isin(['D', 'DR', 'SD', 'BM', 'LR', 'PR', 'ND', 'PO', 'Jackson']),
            election['Result'].isin(['R', 'W', 'F', 'NR', 'AI', 'SR', 'CU', 'KN', 'AM', 'N', 'Adams', 'Crawford', 'Clay']),
            election['Result'].isin(['GW', 'I', 'SP'])
        ],
        [-1, 1, 0],
        default=0
    )
    # Cumulative results
    election['Result_Cumul'] = election_full.groupby('State').Result_plusminus.cumsum()
    ''')
    st.dataframe(election)

    st.code('''
    import plotly.graph_objects as go
    import streamlit as st  

    # Heatmap      
    fig = go.Figure(data=go.Heatmap(
        z=election.Result_Cumul,
        x=election.Year,
        y=election.State,
        colorscale=[[0, 'rgb(0, 66, 202)'], 
                    [0.25, 'rgb(102, 102, 255)'], 
                    [0.5, 'rgb(230, 230, 250)'], 
                    [0.75, 'rgb(255, 102, 102)'], 
                    [1, 'rgb(216, 24, 35)']],
        hoverinfo='text',
        text=election.hover_text1, 
        hovertemplate='%{text}',
        showscale=False
    ))
    fig.update_yaxes(
        tickvals=election.State,
        ticktext=[f"<span style='color: gold; font-size: 22px;'>{state}</span>" if state == ' National' else state for state in election.State],
    )
    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        height=1000
    )
    st.plotly_chart(fig)
    ''')

# Map
#-------------------------------------------------------------
def page3():
    st.header('Wins by State through the Years')
    st.video(f'{path_data}\\gifvideo.mp4', loop=True)

    st.code('''
    import geopandas as gpd
    import pandas as pd

    # State geometries
    poly_states = gpd.read_file(path_data + '\\cb_2018_us_state_20m.shp')
    poly_states['NAME'] = np.where(poly_states.NAME == 'District of Columbia', 'D.C.', poly_states.NAME)
    poly_states = poly_states[poly_states.GEOID != '72'].copy()
    election_gdf = pd.merge(poly_states,
                            election[election.State != ' National'],
                            left_on='NAME',
                            right_on='State',
                            how='left')
    ''')
    st.dataframe(election_gdf)
    st.code('''
    import plotly.graph_objects as go
    import matplotlib.gridspec as gridspec
    import streamlit as st
    from PIL import Image

    for i in election_gdf.Year.unique():
        df = election_gdf[(election_gdf.Year == i)].copy()

        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 5, 1])
        ax = [fig.add_subplot(gs[i]) for i in range(3)] 
        ax[0] = fig.add_subplot(gs[0])
        ax[1] = fig.add_subplot(gs[1])
        ax[2] = fig.add_subplot(gs[2])

        # Alaska and Hawaii
        df0 = df[df.State.isin(['Alaska', 'Hawaii'])].copy()
        df0.plot(
            color=df0.color,
            legend=False,
            ax=ax[0],
            edgecolor='black'
        )
        
        # Continental US
        df1 = df[~df.State.isin(['Alaska', 'Hawaii'])].copy()
        df1.plot(
            color=df1.color,
            legend=False,
            ax=ax[1],
            edgecolor='black'
        )
            
        plt.tight_layout()
        plt.savefig(f"{path_data}\\map_images\\choropleth_map_{i}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
            
        # To GIF
        gifimages = [Image.open(path_data + '\\images\\' + image) for image in image_list]
        gifimages[0].save(path_data + '\\election.gif', save_all=True, append_images=gifimages[1:], duration=175, loop=0)
            
        st.video(path_data + '\\election.gif')
    ''')

pg = st.navigation([st.Page(page0, title='Extract Data'), st.Page(page1, title='Wins by State'), st.Page(page2, title="Cumulative Left vs. Right Wins by State"), st.Page(page3, title='Wins by State through the Years')])
pg.run()


