# Import our Pandas, Streamlit, and Numpy modules
import streamlit as st
import pandas as pd
import numpy as np

# Run this command to run the Streamlit app
# python -m streamlit run dashboard.py

# The different components that we need: Dropdown menu, buttons, sliders, timestamp pickers
# We need fraud alerts, charts, and model predictions that are displayed
# in the Streamlit dashboard

# Import the pickle module - used to serialize and deserialize Python objects
# Serialization = saving a python object to a .pkl file
# Deserialization = loading object into memory
# Useful for trained ML model and loading them without retraining
import pickle

# This is our module which helps us interact with the operating system
import os

# Now we are going to get the io module to help us read and write files
import io

# We are going to import plotly for our visualizations
# plotly is a graphing library that makes interactive, publication-quality graphs online
# It is used for creating interactive visualizations in Python
# We will use plotly.express for quick and easy visualizations
import plotly.express as px
import plotly.graph_objects as go

# Now we are going to load our machine learning pipeline model to our
# Streamlit app
# Make sure that your pkl file is greater than 0 KB so that it's not an empty file

# We need our path for this project directory

# os.getcwd() - this is the method for getting current working directory

model_path = os.path.join(os.getcwd(), 'fraud_detection.pkl')

# We need to check to see if the file exists before loading
if os.path.exists(model_path):
    # We will use the with open path then
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success('Our model is successfully loaded')
else:
    st.error('Our model doesnt exist')
    st.stop()

# Now we are going to get our data from our machine learning pipeline
# We have to load our CSV file that we created for our machine learning pipeline

# Run our model now to get our predictions

# Create our title for our Streamlit dashboard
st.title("FraudSniper")
st.write("Hello, welcome to our Streamlit dashboard")

# Creating other headers titles as such as Overview, Filtered Alerts, and Model Predictions
st.header("Overview")
st.subheader("Filtered Alerts")
st.subheader("Model Predictions")

# Adding our sidebar section
st.sidebar.header("FraudSniper Dashboard")

# Now we are going to add our 3 filter widgets

# First we need create a dataframe by bringing in our creditcard.csv
df = pd.read_csv('creditcard.csv')

# We need to create our risk score in our dataframe using our numpy module
df['risk_score'] = np.random.randint(0, 101, size=len(df))

# Now we are creating a function for assigning severity levels to risk scores
# This is the reverse logic of the risk score mapping
def get_severity(score):
    if score <= 33:
        return 'Low'
    elif score <= 66:
        return 'Medium'
    else:
        return 'High'

df['Severity'] = df['risk_score'].apply(get_severity)

# We can also create a flag filter for our risk score
df['Flag'] = df['Severity'].apply(lambda x: '🔴' if x == 'High' else ('🟠' if x == 'Medium' else '🟢'))

# We can create a map for the risk score to numerical values
risk_mapping = {
    "Low": (0, 33),
    "Medium": (34, 66),
    "High": (67, 100)
}

# Select risk score filter from dropdown menu in sidebar
risk_score = st.sidebar.selectbox("Select Risk Score", ["Low", "Medium", "High"])

# Get your numerical values for the risk score
min_risk, max_risk = risk_mapping[risk_score]

# 2. Timestamp picker for date range
date_range = st.sidebar.date_input("Select Date Range")

# Creating a filter for our alerts based on the date range for right now
# This is our placeholder for the filtered alert timestamp

# First we need to write our DataFrame and convert it into proper datetime format

# This converts seconds from our Time DataFrame into time deltas (total duration)
df['Time'] = pd.to_timedelta(df['Time'], unit='s') + pd.Timestamp('2025-07-29')

# This is a final checkup to make sure that this is a timestamp
df['Time'] = pd.to_datetime(df['Time'])

# First we need to create our DataFrame for the Class and Amount Column,
# since we are checking its binary values for 1 for fraud

df['Class'] = np.random.randint(0, 2, size=len(df))
df['Amount'] = np.random.randint(0, 3000, size=len(df))

# First we will create our amount range slider
amount_range = st.sidebar.slider("Select Amount Range", min_value=0, max_value=3000, value=(0, 2000))

# We now will create the logic behind our flagged anomalies
flagged_anomalies = st.sidebar.checkbox("Flagged Anomalies")

# Now chain all filters properly: so when filtering our dataframe,
# filtering everything together as opposed to one at a time

filtered_df = df[
    (df['Severity'] == risk_score) &
    (df['Amount'] >= amount_range[0]) &
    (df['Amount'] <= amount_range[1])
]

#This is our final step to create edge cases for our model in our main Streamlit app
# Now we are going to create edge cases for our model and feedback messages

#For example, if the filtered Datarame is empty or if we have null values
#in our DataFrame, we will display an error message
if filtered_df.empty:
    st.warning("No transactions match the selected filters.")

if df.isnull().values.any():
    st.warning("Warning: Data contain null values. Some of the filters provided may not work")

# This checks if we have selected a valid date range and if we select it
# it will be a list/tuple of two dates

# This also unpacks our tuple into the 2 dates and
# filters it to include rows where Time is within that date range
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['Time'] >= start_date) & (filtered_df['Time'] <= end_date)]

# Then we create the conditional statements for flagged anomalies,
if flagged_anomalies:
    # We are now creating the filtered df for our Class DataFrame
    filtered_df = filtered_df[filtered_df['Class'] == 1]
    st.write(f"Show {len(filtered_df)} fraud cases in amount range {amount_range[0]} - {amount_range[1]}")
else:
    st.write(f"Showing all transactions in amount range {amount_range[0]} - {amount_range[1]}")

#First we create audit log  dummy dataframes for splunk rule trigger
#We do this so that we can send this data to Splunk HEC(HTTP Event Collector)
if 'SplunkRuleTriggered' not in filtered_df.columns:
    splunk_rules = ['Rule_A', 'Rule_B', 'Rule_C','None']
    filtered_df['SplunkRuleTriggered'] = np.random.choice(splunk_rules, size=len(filtered_df))


#the requests module is used to make HTTP requests
import requests

# the json module is used to parse JSON data
import json

# THis is our Splunk HEC(HTTP Event Collectior) settings
SPLUNK_HEC_URL = "https://127.0.0.1:8088/services/collector"
SPLUNK_TOKEN = "0d5acc53-7402-4815-a6a9-02442afb459b"

#This is our fucntion to send data to Splunk HEC
#We will create a batch size to send our data in chunks
#This is useful for large datasets to avoid overwhelming the server
def send_to_splunk(events, token=SPLUNK_TOKEN, batch_size=1000):
    #This creates a HTTP header and authenticates requests to Splunk with your token
    headers  = {
        "Authorization": f"Splunk {token}"
    }
    #We will now show the total number of events we are sending to Splunk
    total_events = len(events)
    num_batches = (total_events // batch_size) + (1 if total_events % batch_size != 0 else 0)

    #We will show progress text and bar for our batches being sent to Splunk
    progress_text = "Sending flagged anomalies to Splunk"
    progress_bar = st.progress(0, text=progress_text)

    #We will now iterate through our events in batches
    for i in range(num_batches):
        batch = events[i * batch_size:(i + 1) * batch_size]
        #This is the actual data we are sending to Splunk
        # This event is a dictionary that contains our data
        # Tells Splunk that we are sending JSON data
        payload = ""
        for event in batch:
            payload += json.dumps({
             "event": event,
             "sourcetype": "_json",
             }) + "\n"

        # We are making a POST(Posting data to a server) request to the HTTP Event Collector endpoint
        #verify=False is used to skip SSL certification verification - used with self signed certs
        try:
            response = requests.post(SPLUNK_HEC_URL, headers=headers, data=payload, verify=False)
            #Check if our response is successful
            if response.status_code != 200:
                st.warning(f"Batch {i+1}/{num_batches} failed: {response.status_code} - {response.text}")
        #Catch any exceptions that occur during our request
        except Exception as e:
            st.error(f"Batch {i+1}/{num_batches} failed: {e}")
         
        # Update the progress bar
        progress_bar.progress((i + 1) / num_batches, text=progress_text)

    #We will show a success message that our data was sent to Splunk
    st.success("Data successfully sent to Splunk")



# Now we are going to send our flagged anomalies to Splunk
# We will iterate through each row in the filtered DataFrame
if flagged_anomalies:
  if st.sidebar.button("Send Flagged Anomalies to Splunk"):
    # Create an array to hold our events
    events_to_send = []
    for _, row in filtered_df.iterrows():
        try:
            # Builds a dictionary for each row with the timestamp
            # We will clean each row before sending and convert
            # This is to prevent so that no NaNs or None's get into the JSON
            event = {
                "timestamp": str(row['Time']) if pd.notna(row['Time']) else "unknown",
                "risk_score": float(row['risk_score']) if pd.notna(row['risk_score']) else 0.0,
                "severity": str(row['Severity']) if pd.notna(row['Severity']) else "N/A",
                "amount": float(row['Amount']) if pd.notna(row['Amount']) else 0.0,
                "splunk_rule": str(row['SplunkRuleTriggered']) if pd.notna(row['SplunkRuleTriggered']) else "None"
            }
            # We will append each event to our events array
            events_to_send.append(event)
        except Exception as e:
            # Skip rows that cause issues and notify the user
            st.warning(f"Skipping malformed row due to error: {e}")
    
    # Now we will send our events to Splunk in batches
    send_to_splunk(events_to_send)


# This displays our filtered table where we can display our fraud anomalies
st.dataframe(filtered_df)

# Now we are going to create our styling function for row highlighting
def highlight_rows(row):
    if row['Severity'] == 'High':
        return ['background-color: red'] * len(row)
    elif row['Severity'] == 'Medium':
        return ['background-color: yellow'] * len(row)
    else:
        return ['background-color: green'] * len(row)

#Now we are going to create an export and download button for our filtered data

# Converted our filtered DataFrame to CSV format

# Creates an in-memory buffer to hold CSV data
csv_buffer = io.StringIO()

# Write our filtered DataFrame to the buffer in CSV format
filtered_df.to_csv(csv_buffer, index=False)

#extracts to entire of that in-memory buffer and stores it in csv_data
csv_data = csv_buffer.getvalue()

# Create the download button for our filtered data
st.download_button(label="Downloaded Filtered Data as CSV",
                   data=csv_data,
                   file_name='filtered_data.csv',
                   #Tells us that the file tupe is CSV so it handles it properly
                   mime='text/csv')

#This is where we are adding out audit trail module

#Built audit log for Splunk rule triggered
audit_log = filtered_df[['Time','risk_score', 'SplunkRuleTriggered']].copy()
audit_log.rename(columns={'Time': 'Timestamp', 'risk_score': 'Score'}, inplace=True)

# Show audit log on dashboard
st.dataframe(audit_log)

# Audit trail CSV
csv_buffer_audit = io.StringIO()
audit_log.to_csv(csv_buffer_audit, index=False)
csv_data_audit = csv_buffer_audit.getvalue()

st.download_button(
    label="Download Audit Trail as CSV",
    data=csv_data_audit,
    file_name='audit_trail.csv',
    mime='text/csv'
)

# Audit trail JSON
json_data_audit = audit_log.to_json(orient='records', lines=False)

st.download_button(
    label="Download Audit Trail as JSON",
    data=json_data_audit,
    file_name='audit_trail.json',
    mime='application/json'
)


#Now we are going to create the summary stats in overview section
st.header("Summary Statistics Overview")

#We can calculate the total number of transacted through the length
# of our filtered DataFrame
total_transactions = len(filtered_df)

# Then we need to filter alls the rows where our Class is 1
# The .shape[0] returns the count of those rows
# This will return the total # of flagged anomalies
flagged_count = filtered_df[filtered_df['Class'] == 1].shape[0]



# we can now count the number of severity levels we have
# Counts the number of occurrences at each severity level in a series
# Example Visual:
# High 50
# Medium 30
# Low 20
#
severity_counts = filtered_df['Severity'].value_counts()

# Now we are going to have a metric widgets for show
#  the total transactions & flagged anomalies
st.metric("Total Transactions", total_transactions)
st.metric("Flagged Anomalies", flagged_count)

# Now we and going to use the st.write() methods to render
# our dataframes for severity counts
#This converts the severity counts from a Series to a DataFrame
# and displays it into a table format
st.write("Severity Counts:")
st.write(severity_counts.to_frame())

#Now we are going to create the visualizations for our model predictions

#We are going to create a time series plot for the # of flagged anomalies
#Markers are used to highlight points on the line chart

#First we need to created a DataFrame for the flagged anomalies over time
# We will now created a filtered DataFrame for Dates by using the Time column
filtered_df['Date'] = filtered_df['Time'].dt.date

flagged_count_df = (
    #Filters the DataFrame to include only flagged anomalies
    filtered_df[filtered_df['Class'] == 1]
    #We are grouping the filtered fraud by the Date column
    .groupby('Date')
    #We are going to count how many rows we have for each date
    .size()
    #Now we have a Series, we will reset index to convert it to a DataFrame
    .reset_index(name='Flagged Count')
)

flagged_time_series = px.line(flagged_count_df, x='Date', y='Flagged Count',
                              title='# of Flagged Fraud Anomalies Over Time',
                              markers=True)
# Now we will display our flagged time series plot
st.plotly_chart(flagged_time_series)

#Now we are going to create bar chart for the severity levels

severity_count_df = (
    #Filter the DataFrame to include only severity count
    filtered_df['Severity'].value_counts()
    #We are grouping the filtered fraud by their Severity column
    #Now we have a Series, we will reset index to convert it to a DataFrame
    .reset_index(name='Severity Level Count'))

#Now We are going to rename the columns
severity_count_df.columns = ['Severity Level', 'Severity Level Count']

# Now we are going to plot our bar graph
sev_level_bar = px.bar(severity_count_df,
                       x='Severity Level',
                       y='Severity Level Count',
                       title='# of Severity Levels Over Time')

# Now we can show out plot
st.plotly_chart(sev_level_bar)

#Now we are going to create our KPIs(Key Performance Indicators)

#Total anomalies flagged

#Counts our amount with shape[0]
total_anomalies = df[df['Class']== 1].shape[0]

#Top risk score - finds maximum severity score across all records
top_risk_score = filtered_df['Severity'].mode()[0] if not filtered_df.empty else 'Low'
#I'm going to create a mapping for these severity string since
#the indicator values expects an int or float(something that is numeric)
severity_count_map = {'Low': 1, 'Medium': 2, 'High': 3}
top_risk_score_numeric = severity_count_map.get(top_risk_score, 0)

# Let's create our KPIs now  with Plotly FigureWeight
my_kpis = go.Figure()

# The add_trace() allows us the add a new visual element to Plotly figure

# The go.Indicator is a special Plotly object cleans boxes that show
# numbers, deltas, gauges
my_kpis.add_trace(go.Indicator(
    #Mode tells you for this case we are just showing the number
    mode="number",
    # value tells us the actual number that's displayed
    value=total_anomalies,
    #title, this adds label so people know what this gauge does
    title= {'text': 'Total Anomalies'},
    #domain allows you to control the positions of your dashboard gauge
    # with your x and y coordinates
    domain={'x': [0, 0.3], 'y': [0.5,1]}))

my_kpis.add_trace(go.Indicator(
    #Mode tells you for this case we are just showing the number
    mode="number",
    # value tells us the actual number that's displayed
    value=top_risk_score_numeric,
    #title, this adds label so people know what this gauge does
    title={'text': f'Top Risk Score ({top_risk_score})'},
    #domain allows you to control the positions of your dashboard gauge
    # with your x and y coordinates
    domain={'x': [0.7, 1], 'y': [0.5,1]}))

#Now we will show the entire thing as our dashboard
my_kpis.update_layout(title='My Fraud Detection KPIs')
st.plotly_chart(my_kpis)