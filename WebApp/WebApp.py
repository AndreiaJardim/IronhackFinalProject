#To run the code: python -m streamlit run WebApp.py 

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib  # To load the pickled model


# Set the Streamlit page layout to be wider
st.set_page_config(layout="wide")

# Load your machine learning model
model = joblib.load("C:/Users/jardi/Desktop/Ironhack/FINAL PROJECT/xgboost_regressor_model.pkl")

# Load the MinMaxScaler
normalizer = joblib.load("C:/Users/jardi/Desktop/Ironhack/FINAL PROJECT/min_max_scaler.pkl")

# Define a custom CSS style for the background color
background_color_style = """
<style>
body {
    background-color: #F9F3E4;
}
</style>
"""

# Add a header image
st.image('wineBottles.png', use_column_width=True)

# Homepage
st.title("GRAPES TO GRADES!")

# Homepage
st.write("Choose a page:")

# Place buttons side by side
col1, col2, col3 = st.columns(3)

# Initialize session state variables
if 'HOME' not in st.session_state:
    st.session_state.Home = False
if 'WINEMAKER' not in st.session_state:
    st.session_state['WINEMAKER'] = False
if 'WINE PORTFOLIO' not in st.session_state:
    st.session_state['WINE PORTFOLIO'] = False

if col1.button("HOME", key="home_button"):
    # Set all page flags to False to return to the homepage
    st.session_state.Home = True
    st.session_state['WINEMAKER'] = False
    st.session_state['WINE PORTFOLIO'] = False

if col2.button("WINEMAKER", key="page_1_button"):
    st.session_state['WINEMAKER'] = True
    st.session_state.Home = False  # Remove Home state if it exists
    st.session_state['WINE PORTFOLIO'] = False  # Remove Page 2 state if it exists

if col3.button("WINE PORTFOLIO", key="page_2_button"):
    st.session_state['WINE PORTFOLIO'] = True
    st.session_state.Home = False  # Remove Home state if it exists
    st.session_state['WINEMAKER'] = False  # Remove Page 1 state if it exists

# Page 1
def page_1():
    wine = pd.read_excel("C:\\Users\\jardi\\Desktop\\Ironhack\\FINAL PROJECT\\wine_characteristics.xlsx")
    
    # Filter out rows where 'Grapes_y' is not NaN and is not a float
    wine = wine[~wine['Grapes_y'].apply(lambda x: isinstance(x, float))]
    
    # Split the combined grape varieties into a list of individual grapes
    wine['Grapes_y'] = wine['Grapes_y'].str.split(', ')
    
    # Create a set of unique grape varieties
    unique_grapes = set()
    for grapes_list in wine['Grapes_y']:
        unique_grapes.update(grapes_list)

    st.title("Be a Winemaker")
    st.write("Below, you can input wine features and use our predictive model to estimate the average wine rating.")
    st.header('Wine Features')
    
    col1, col2 = st.columns(2)
    with col1:
        WineName = st.text_input('Give your Wine a name:', '')
        WineType = st.selectbox('What type of wine do you want to produce?', ('','Red', 'White', 'Sparkling', 'Rosé','Dessert', 'Fortified'))
        if WineType:
            pass
        else:
            pass
        #st.write('You selected:', WineType)
        RegionOption = st.selectbox('Where would you like to plant the grapes?', ('','Porto', 'Beja', 'Castelo Branco', 'Lisboa', 'Angra do Heroísmo', 'Funchal'))
        if WineType:
            pass
        else:
            pass

    with col2:
        grapes = st.multiselect('What kind of grapes do you want to have on your wine?', (unique_grapes))
        
        price = st.number_input('Enter the expected price for your wine:', min_value=0.0, max_value=2000.0)
        if not isinstance(price, (float, int)):
            st.warning("Please enter a numeric value for the price.")
        allergens = st.selectbox('Your wine will contain allergens?', ('','Without allergens', 'Contains sulfites','Contains sulfites, egg allergens, milk allergens',
       'Contains sulfites, egg allergens'))
        #ratings = st.slider('Select the amount of ratings necessary:', 0.0, 10000.0)

    if st.button('Predict Wine Rating'):
        if WineType and RegionOption and grapes and price and allergens:
            # Collect user input data (WineType, RegionOption, allergens, grapes, price, ratings)
            user_input = {
                'Year': 2023,
                'Allergens': allergens,
                'Wine Type': WineType,
                'Ratings': 10000,
                'Price':price,
                'Grapes': grapes,
                'Region': RegionOption         

            }

            def preprocess_data(user_input, unique_grapes, normalizer):
                # Create a DataFrame for user input
                user_input_df = pd.DataFrame({
                    'Year': user_input['Year'],
                    'Ratings': user_input['Ratings'],
                    'Price': user_input['Price'],
                    'Region2': [user_input['Region']],
                    'Allergens__Contains sulfites, egg allergens': 0,
                    'Allergens__Contains sulfites, egg allergens, milk allergens': 0,
                    'WineType_Fortified': 0,
                    'WineType_Red': 0,
                    'WineType_Rosé': 0,
                    'WineType_Sparkling': 0,
                    'WineType_White': 0,
                    'Grape_Gewürztraminer': 0, 'Grape_Primitivo': 0, 'Grape_Encruzado': 0, 'Grape_Moscatel': 0, 'Grape_Malvasia Fina': 0,'Grape_Baroque': 0, 'Grape_Malvasia': 0, 'Grape_Tamarez': 0, 'Grape_Verdelho': 0, 
                    'Grape_Viosinho': 0, 'Grape_Alvarelhao': 0,'Grape_Baga': 0, 'Grape_Touriga-Fêmea': 0, 'Grape_Tinta Barroca': 0, 'Grape_Tannat': 0, 'Grape_Rabigato': 0, 'Grape_Rabo de Ovelha': 0, 'Grape_Cerceal Branco': 0,
                    'Grape_Assario Branco': 0, 'Grape_Folgasao': 0, 'Grape_Grenache': 0, 'Grape_Touriga Franca': 0, 'Grape_Tinta Caiada': 0, 'Grape_Arinto de Bucelas': 0, 'Grape_Moscatel Roxo': 0, 'Grape_Alicante Bouschet': 0,
                    'Grape_Sémillon': 0, 'Grape_Tinta Roriz': 0, 'Grape_Marsanne': 0, 'Grape_Vermentino': 0, 'Grape_Trousseau': 0, 'Grape_Moscatel de Setúbal': 0, 'Grape_Petit Verdot': 0, 'Grape_Viognier': 0, 
                    'Grape_Shiraz/Syrah': 0,'Grape_Tinta Francisca': 0, 'Grape_Tinta Cão': 0, 'Grape_Arinto dos Açores': 0, 'Grape_Jampal': 0, 'Grape_Maria Gomes': 0, 'Grape_Riesling': 0, 'Grape_Tinta Miuda': 0,
                    'Grape_Gouveio': 0, 'Grape_Gouveio Real': 0, 'Grape_Fonte Cal': 0,'Grape_Loureiro': 0, 'Grape_Azal': 0, 'Grape_Sauvignon Blanc': 0, 'Grape_Codega de Larinho': 0, 'Grape_Sercial': 0,'Grape_Pinot Gris': 0, 
                    'Grape_Bastardo': 0, 'Grape_Alicante Ganzin': 0, 'Grape_Fernao Pires': 0, 'Grape_Pinot Noir': 0,'Grape_Chardonnay': 0, 'Grape_Cinsault': 0, 'Grape_Tinto Cao': 0, 'Grape_Trincadeira das Pratas': 0, 
                    'Grape_Touriga Nacional': 0,'Grape_Casculho': 0, 'Grape_Tinta Carvalha': 0, 'Grape_Moreto': 0, 'Grape_Alfrocheiro': 0, 'Grape_Greco Bianco': 0,'Grape_Pinot Meunier': 0, 'Grape_Sercialinho': 0, 
                    'Grape_Boal Branco': 0, 'Grape_Galego Dourado': 0, 'Grape_Souzao': 0,'Grape_Bastardo Magarachsky': 0, 'Grape_Donzelinho Tinto': 0, 'Grape_Aragonez': 0, 'Grape_Cornifesto': 0, 'Grape_Perrum': 0,
                    'Grape_Tempranillo': 0, 'Grape_Manteudo': 0, 'Grape_Marufo': 0, 'Grape_Albariño': 0, 'Grape_Cabernet Franc': 0,'Grape_Moscatel Galego': 0, 'Grape_Jaen': 0, 'Grape_Grossa': 0, 'Grape_Aspiran Bouchet': 0, 
                    'Grape_Alfrocheiro Preto': 0,'Grape_Cabernet Sauvignon': 0, 'Grape_Tinta Amarela': 0, 'Grape_Grand Noir': 0, 'Grape_Trajadura': 0, 'Grape_Terrantes do Pico': 0,'Grape_Trincadeira': 0, 'Grape_Avesso': 0, 
                    'Grape_Malvasia Preta': 0, 'Grape_Merlot': 0, 'Grape_Chasselas': 0,'Grape_Touriga Francesa': 0, 'Grape_Diagalves': 0, 'Grape_Antão Vaz': 0, 'Grape_Terrantez': 0, 'Grape_Petit Manseng': 0,'Grape_Bical': 0, 
                    'Grape_Donzelinho Branco': 0, 'Grape_Barcelo': 0, 'Grape_Vinhão': 0, 'Grape_Mourisco': 0, 'Grape_Rufete': 0,'Grape_Alvarinho': 0, 'Grape_Castelao': 0, 'Grape_Síria': 0, 'Grape_Roupeiro': 0,
                    'Temperature':0,'Precipitation':0,'Region2_Angra do Heroísmo':0,'Region2_Beja':0,'Region2_Castelo Branco':0,'Region2_Funchal':0,'Region2_Lisboa':0,	'Region2_Porto':0   
                })

                # Handle Allergens columns
                if user_input['Allergens'] == 'Contains sulfites, egg allergens':
                    user_input_df['Allergens__Contains sulfites, egg allergens'] = 1
                elif user_input['Allergens'] == 'Contains sulfites, egg allergens, milk allergens':
                    user_input_df['Allergens__Contains sulfites, egg allergens, milk allergens'] = 1

                # Handle WineType columns
                wine_type_mapping = {
                    'Red': 'WineType_Red',
                    'Dessert': 'WineType_Fortified',
                    'Fortified': 'WineType_Fortified',
                    'Rosé': 'WineType_Rosé',
                    'Sparkling': 'WineType_Sparkling',
                    'White': 'WineType_White'
                }
                wine_type = user_input['Wine Type']
                if wine_type in wine_type_mapping:
                    user_input_df[wine_type_mapping[wine_type]] = 1

                # Iterate through grape columns and set values based on user's selected grapes
                for grape_column in user_input_df.columns:
                    if grape_column.startswith('Grape_'):
                        grape_name = grape_column[len('Grape_'):]
                        if grape_name in user_input['Grapes']: # Checks if the grape_name extracted from the column name is present in the list of grapes that the user has selected (user_input['Grapes'])
                            user_input_df[grape_column] = 1
                        else:
                            user_input_df[grape_column] = 0

                # Iterate through region columns and set values based on user's selected grapes
                for region_column in user_input_df.columns:
                    if region_column.startswith('Region2_'):
                        region_name = region_column[len('Region2_'):]
                        if region_name in user_input['Region']: # Checks if the region2_name extracted from the column name is present in the list of grapes that the user has selected (user_input['Grapes'])
                            user_input_df[region_column] = 1
                        else:
                            user_input_df[region_column] = 0

                # Load average temperature data (AvgTemp) and total precipitation data (TotPrecipitation)
                AvgTemp = pd.read_excel('C:\\Users\\jardi\\Desktop\\Ironhack\\FINAL PROJECT\\Temperature_data\\average_temperaturePT.xlsx')
                TotPrecipitation = pd.read_excel('C:\\Users\\jardi\\Desktop\\Ironhack\\FINAL PROJECT\\Temperature_data\\total_precipitationPT.xlsx')

                # Filter AvgTemp and TotPrecipitation DataFrames based on user input region and year
                avg_temp_column = user_input['Region']
                year_column = user_input['Year']

                user_input_df['Temperature'] = AvgTemp[avg_temp_column][AvgTemp['Year'] == year_column].values
                user_input_df['Precipitation'] = TotPrecipitation[avg_temp_column][TotPrecipitation['Year'] == year_column].values

                #Drop column Region2 because it is not part of the columns of the model
                user_input_df = user_input_df.drop(columns=['Region2'])

                # Normalize the entire DataFrame using the provided normalizer
                user_input_df = pd.DataFrame(normalizer.transform(user_input_df), columns=user_input_df.columns)

                return user_input_df

            # Preprocess user input data using the function from preprocessing_module
            preprocessed_user_input = preprocess_data(user_input, unique_grapes, normalizer)

            # Make predictions using your machine learning model
            predicted_rating = model.predict(preprocessed_user_input)

            # Display the predicted rating to the user in bold
            st.markdown(f"<strong>Predicted Average Wine Rating:</strong> {predicted_rating[0]:.2f}", unsafe_allow_html=True)

            # Add a space or separator line
            st.write("")  # Empty line

            # Create a section or container for displaying wine characteristics
            st.subheader("Your Wine Characteristics")

            # Create a dictionary to store the wine characteristics
            wine_characteristics = {
                "Wine Name": WineName,
                "Wine Type": WineType,
                "Region": RegionOption,
                "Grapes": ', '.join(grapes),
                "Expected Price": f"€{price:.2f}",
                "Allergens": allergens,
                "Predicted Average Wine Rating": f"{predicted_rating[0]:.2f}",
            }

            # Create a DataFrame with "Characteristic" and "Value" columns
            wine_df = pd.DataFrame(wine_characteristics.items(), columns=["Characteristic", "Value"])

            # Display the wine characteristics as a table with custom CSS
            st.markdown(
                """
                <style>
                table {
                    width: 10%;
                    table-layout: fixed;
                }
                th {
                    word-wrap: break-word;
                }
                td {
                    word-wrap: break-word;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )


            # Display the table with "Characteristic" and "Value" columns and hide the index
            st.table(wine_df.set_index('Characteristic'))
        else:
            st.warning("Please fill in all required fields before predicting wine rating.")



        
# Page 2
def page_2():
    st.title("Wine Portfolio")
    st.write("Explore Portugal's Wine Portfolio:")
    
    # Define the HTML code for your Power BI report with a CSS class for styling
    report_html = """
    <div class="custom-iframe-container">
        <iframe title="WinePortfolio" width="1300" height="1060" src="https://app.powerbi.com/view?r=eyJrIjoiMGFhY2ViMDgtMjI1Ni00MmNmLTk0MzQtYjllZDJkNmE0OWI1IiwidCI6ImJmNDUwYjNiLTdlYTQtNGU1NS1hZDAyLTYzMTNjZDM0ZDBiMSIsImMiOjl9" frameborder="0" allowFullScreen="true"></iframe>
    </div>
    <style>
        /* Define the CSS class for custom iframe container */
        .custom-iframe-container {
            height: 1500px; /* Adjust the height as needed */
            overflow: auto; /* Add scrollbars if the content exceeds the height */
        }
    </style>
    """

    # Use st.components.v1.html to embed the HTML code
    st.components.v1.html(report_html, height=1600)  # Adjust the height parameter as needed


# Check the selected page and display accordingly
if st.session_state.Home:
    st.title("Welcome to 'Grapes to Grades', the Wine Analysis App!")
    st.write("""
    This app is designed to help you explore and analyze Portuguese wines. Whether you're a wine enthusiast, a winemaker, or simply curious about wine ratings, you've come to the right place.

    **How it works:**

    - **Winemaker Page:** Are you a winemaker looking to predict the average rating of your wines for the upcoming year? Head over to the Winemaker Page, where you can input wine features and use our predictive model to estimate the average wine rating.

    - **Wine Portfolio Page:** Interested in exploring a portfolio of Portuguese wines? Check out the Wine Portfolio Page, where you can find interactive insights and visualizations of our curated collection of Portuguese wines.

    **About the Predictive Model:**

    Our predictive model is trained on a dataset of Portuguese wines, making it highly accurate for wines from this region. Whether you're a winemaker looking to improve your wine's rating or a wine lover interested in discovering new favorites, our model can provide valuable insights.

    **Explore and Enjoy:**

    We invite you to explore the different pages of this app and make the most of the wine-related resources we've provided. Cheers to a journey of wine discovery!

    Enjoy your time with the Wine Analysis App!
    """)

elif st.session_state['WINEMAKER']:
    page_1()
elif st.session_state['WINE PORTFOLIO']:
    page_2()


