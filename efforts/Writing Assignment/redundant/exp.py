goal_code = 'Goal Code'
goal_lbl = 'Goal Label'
goal_desc = 'Goal Description '
tgt_code = 'Target Code'
tgt_desc = 'Target Description'
ind_code = 'Indicator Code'
ind_ref = 'Indicator Reference'
ind_desc = 'Indicator Description'
series_rel = 'Series Release'
tags = 'Tags'
series_code = 'Series Code'
series_desc = 'Series Description'
geo_code = 'Geographic Area Code'
geo_name = 'Geographic Area Name'
geo_lvl = 'Geographic Area Level'
geo_parent_code = 'Parent Geographic Area Code'
geo_parent_name = 'Parent Geographic Area Name'
geo_type = 'Geographic Area Type'
coord_x = 'X'
coord_y = 'Y'
iso_code = 'ISO Code'
is_un = 'Is UN Member'
has_profile = 'Has Country Proile'
unit_mult = 'Unit Multiplier'
unit_code = 'Units Code'
unit_desc = 'Units Desc'
ts_id = 'Time Series Id'
ts_keys = 'Time Series Keys'
yrs_avail = 'Number of Years Available'
yr_earliest = 'Earliest Year Available'
yr_latest = 'Latest Year Available'
avail_yrs = 'Available Years'

# Time series values (can also be stored in a loop or dictionary if needed)
val_1967 = 'Value 1967'
val_1968 = 'Value 1968'
val_1969 = 'Value 1969'
val_1970 = 'Value 1970'
val_1971 = 'Value 1971'
val_1972 = 'Value 1972'
val_1973 = 'Value 1973'
val_1974 = 'Value 1974'
val_1975 = 'Value 1975'
val_1976 = 'Value 1976'
val_1977 = 'Value 1977'
val_1978 = 'Value 1978'
val_1979 = 'Value 1979'
val_1980 = 'Value 1980'
val_1981 = 'Value 1981'
val_1982 = 'Value 1982'
val_1983 = 'Value 1983'
val_1984 = 'Value 1984'
val_1985 = 'Value 1985'
val_1986 = 'Value 1986'
val_1987 = 'Value 1987'
val_1988 = 'Value 1988'
val_1989 = 'Value 1989'
val_1990 = 'Value 1990'
val_1991 = 'Value 1991'
val_1992 = 'Value 1992'
val_1993 = 'Value 1993'
val_1994 = 'Value 1994'
val_1995 = 'Value 1995'
val_1996 = 'Value 1996'
val_1997 = 'Value 1997'
val_1998 = 'Value 1998'
val_1999 = 'Value 1999'
val_2000 = 'Value 2000'
val_2001 = 'Value 2001'
val_2002 = 'Value 2002'
val_2003 = 'Value 2003'
val_2004 = 'Value 2004'
val_2005 = 'Value 2005'
val_2006 = 'Value 2006'
val_2007 = 'Value 2007'
val_2008 = 'Value 2008'
val_2009 = 'Value 2009'
val_2010 = 'Value 2010'
val_2011 = 'Value 2011'
val_2012 = 'Value 2012'
val_2013 = 'Value 2013'
val_2014 = 'Value 2014'
val_2015 = 'Value 2015'
val_2016 = 'Value 2016'
val_2017 = 'Value 2017'
val_2018 = 'Value 2018'
val_2019 = 'Value 2019'

val_latest = 'Latest Value'
footnotes = 'Footnotes'
nature = 'Nature'
obj_id = 'ObjectId'
x2 = 'x2'
y2 = 'y2'

target_countries = ['Japan', 'United States of America', 'United Kingdom of Great Britain and Northern Ireland', 'Germany', 'Canada', 'India', 'China', 'Brazil','Indonesia','South Africa','Haiti','Mozambique','Yemen']

























import pandas as pd
import plotly.express as px
import sys # To check if running in a suitable environment for interactive widgets

# --- Configuration ---
# Specify the path to your CSV file
csv_file_path = 'gender_seats.csv' # <-- Make sure this is the correct path/name for your ALCOHOL data CSV

# Specify the column name that indicates the parent region for each country.
# !!! IMPORTANT: Change this to match the actual column name in YOUR CSV file !!!
region_col = geo_parent_name # <-- Example: might be 'Region', 'Continent', 'Parent Region' etc.

# Specify the column name possibly used for initial filtering (if applicable)
# If you don't need this filter for the alcohol data, you can leave it commented out.
# filter_col = 'yrs_avail' # <-- Example column name, change if needed
# filter_threshold = 20

# Define the FULL range of years potentially present in your 'Value YEAR' columns
# We will filter the *display* range later
start_year = 2000
end_year = 2022 # <<< ADJUSTED based on request, make sure your data actually has 'Value 2022' if needed

# Define the specific years to focus the visualization on
focus_start_year = 2000
focus_end_year = 2022

# Define the desired fixed point size
marker_point_size = 18 # Adjust as needed
# --- End Configuration ---

# --- Data Loading and Preparation ---
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print("Please make sure the file exists and the path is correct.")
    sys.exit(1) # Exit the script if file not found

# Optional: Check the dataframe columns to verify the region column name and timeline columns
print("Columns in the dataset:")
print(df.columns)
print("-" * 30)

# Verify the specified region column exists
if region_col not in df.columns:
    print(f"Error: The specified region column '{region_col}' does not exist in the CSV.")
    print(f"Available columns are: {list(df.columns)}")
    print("Please update the 'region_col' variable in the script.")
    sys.exit(1)

# --- Optional Initial Filtering ---
# Uncomment and adjust this block if you need an initial filter based on some criteria
# if filter_col in df.columns:
#     print(f"Applying filter: Keep rows where '{filter_col}' >= {filter_threshold}")
#     df = df[df[filter_col] >= filter_threshold].copy() # Use .copy() to avoid SettingWithCopyWarning
#     print(f"Data shape after filtering: {df.shape}")
# else:
#     print(f"Warning: Filter column '{filter_col}' not found. Skipping initial filtering.")
# print("-" * 30)
# --- End Optional Initial Filtering ---

# Define the timeline columns based on the FULL year range
timeline_cols = [f'Value {year}' for year in range(start_year, end_year + 1)]

# Check if at least some timeline columns exist
existing_timeline_cols = [col for col in timeline_cols if col in df.columns]
if not existing_timeline_cols:
    print("Error: None of the expected timeline columns (e.g., 'Value 2000', 'Value 2001', ...) were found.")
    print(f"Expected format: 'Value YEAR' for years {start_year} to {end_year}.")
    print(f"Available columns are: {list(df.columns)}")
    sys.exit(1)
elif len(existing_timeline_cols) < len(timeline_cols):
     print(f"Warning: Some expected timeline columns were not found. Using the ones available: {existing_timeline_cols}")

print(f"Using region column: '{region_col}'")
print(f"Using timeline columns: {existing_timeline_cols}")
print("-" * 30)

# Reshape the dataframe from wide to long format
df_long = pd.melt(df,
                  id_vars=[region_col],
                  value_vars=existing_timeline_cols,
                  var_name="Year_Raw",
                  value_name="Alcohol Consumption")

# Clean the "Year" column: remove "Value " prefix and convert to numeric
df_long['Year'] = df_long['Year_Raw'].str.replace('Value ', '', regex=False)
df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')

# Drop rows where conversion failed or where alcohol consumption is missing
df_long.dropna(subset=['Year', 'Alcohol Consumption'], inplace=True)

# Convert Year to integer
df_long['Year'] = df_long['Year'].astype(int)

# Compute the average alcohol consumption for each parent region in each year
df_grouped = df_long.groupby([region_col, "Year"], as_index=False)["Alcohol Consumption"].mean()

# Ensure the data is sorted by Year for smooth animation
df_grouped = df_grouped.sort_values("Year")

print("Sample of processed data (full time range):")
print(df_grouped.head())
print("-" * 30)

# --- Filter Data for the Specific Time Window (2020-2022) ---
print(f"Filtering data for the years: {focus_start_year} to {focus_end_year}")
df_filtered_time = df_grouped[
    (df_grouped['Year'] >= focus_start_year) &
    (df_grouped['Year'] <= focus_end_year)
].copy() # Use .copy() to avoid potential SettingWithCopyWarning later

# Check if any data remains after filtering
if df_filtered_time.empty:
    print(f"Error: No data available for the selected year range {focus_start_year}-{focus_end_year} after processing.")
    print("Check your original data and the year range specified in the configuration.")
    sys.exit(1)

print(f"Data shape after filtering for {focus_start_year}-{focus_end_year}: {df_filtered_time.shape}")
print("Sample of filtered data for plotting:")
print(df_filtered_time.head())
print("-" * 30)
# --- End Filtering ---

# --- Interactive Plotting with Plotly ---
print("Generating interactive plot for the selected time range...")

# Determine the overall range for the y-axis based on the FILTERED data
# This keeps the Y-axis consistent within the animation of the filtered years
y_min = df_filtered_time["Alcohol Consumption"].min() * 0.95 # Add 5% buffer at bottom
y_max = df_filtered_time["Alcohol Consumption"].max() * 1.05 # Add 5% buffer at top

# Determine the range for the x-axis based on the FILTERED data years
# Add a small buffer (e.g., 0.5 years) for better visualization if desired
x_min = df_filtered_time["Year"].min() - 0.5
x_max = df_filtered_time["Year"].max() + 0.5

fig = px.scatter(
    df_filtered_time, # Use the time-filtered dataframe
    x="Year",
    y="Alcohol Consumption",
    color=region_col,           # Color points by region, creates interactive legend
    animation_frame="Year",     # Creates the slider based on filtered Years
    animation_group=region_col, # Keeps track of regions across frames
    hover_name=region_col,      # Show region name on hover
    range_y=[y_min, y_max],     # Fix Y-axis range based on filtered data
    range_x=[x_min, x_max],     # Fix X-axis range to the desired focus years
    title=f"Alcohol Consumption from {focus_start_year} to {focus_end_year} by Parent Region",
    labels={
        "Alcohol Consumption": "Alcohol Consumption (Avg. per Capita)",
        region_col: "Region"
    }
)

# --- Increase Marker Size ---
# Update the marker size for all traces in the figure
fig.update_traces(marker=dict(size=marker_point_size)) # Apply the configured size
# --- End Increase Marker Size ---

# Update layout for better appearance
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Alcohol Consumption (Avg. per Capita)",
    legend_title_text='Region'
)

# Adjust animation speed (optional)
# The slider will now only contain frames for the filtered years (2020, 2021, 2022)
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 700 # ms per frame
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300 # ms transition

# Show the plot
fig.show()

print("Plot generation complete. Check your browser or output cell.")