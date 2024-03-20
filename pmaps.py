import streamlit as st
import pandas as pd
import pickle
import base64

def get_download_link(df, filename):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
    
def processing(df):
    st.text("Step 1: Filtering data for 'PMaps Sales Orientation' section...")
    ques_id_to_drop = [25131, 25132, 25133, 25134, 25140, 25141, 25142, 25143, 25144]
    df = df[df['SectionName'] == 'PMaps Sales Orientation']
    
    st.text("Step 2: Selecting relevant columns...")
    df = df[['CandidateUniqueId', 'EmailAddress', 'OriginalQuestionId', 'OriginalOptionId']]
    
    st.text("Step 3: Dropping specified question IDs...")
    df = df[~df['OriginalQuestionId'].isin(ques_id_to_drop)]
    
    st.text("Step 4: Grouping by candidate and question...")
    df = df.groupby(['EmailAddress', 'OriginalQuestionId']).last().reset_index()
    
    st.text("Step 5: Pivoting the DataFrame...")
    df_pivoted = df.pivot(index='CandidateUniqueId', columns='OriginalQuestionId', values='OriginalOptionId')
    df_pivoted = df_pivoted.fillna(0).astype(int)
    
    st.text("Step 6: One-hot encoding the data...")
    encoded_df = pd.get_dummies(df_pivoted, prefix_sep='_', columns=df_pivoted.columns)
    encoded_df = encoded_df[[col for col in encoded_df.columns if not col.endswith('_0')]]
    
    st.text("Step 7: Creating a new DataFrame with encoded values...")
    new_df = pd.DataFrame(index=df_pivoted.index, columns=encoded_df.columns, dtype=int).fillna(0)
    new_df.update(encoded_df)

    return new_df.copy(deep=True)

def main():
    st.title("PMaps Prediction App")
    model = pickle.load(open('model.pkl', 'rb'))

    st.sidebar.title("Upload Positonal Data")
    
    instructions = """
    Instructions:
    Please provide a CSV file containing the following mandatory columns:

    1. `CandidateUniqueId`: Unique identifier for each candidate.
    2. `EmailAddress`: Email address of the candidate.
    3. `SectionName`: Name of the section in the test.
    4. `OriginalQuestionId`: Identifier for the original question.
    5. `OriginalOptionId`: Identifier for the original option selected by the candidate.

    Ensure that the CSV file is sorted in a positional manner, with the first entry representing the results of the first attempt.
    """
    st.sidebar.markdown(instructions)
    st.sidebar.markdown("Please confirm that you have read and understood the instructions below before proceeding.")
    instructions_confirmed = st.sidebar.checkbox("I confirm that I have read and understood the instructions")
    
    if instructions_confirmed:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            st.sidebar.text("Processing...")
            df = pd.read_csv(uploaded_file)
            processed_df = processing(df)
            
            st.sidebar.text("Making Predictions...")
            preds = model.predict(processed_df)
            probs = model.predict_proba(processed_df)
            probs = list(map(lambda x: str(round(100*max(x),2))+'%', probs))
    
            df_out = processed_df.copy(deep=True)
            df_out['Performance'] = list(map(lambda x: x.split('_')[0] , preds))
            df_out['Attrition'] = list(map(lambda x: x.split('_')[1] , preds))
            df_out['Confidence'] = probs
    
            df_out = df_out[['Performance', 'Attrition', 'Confidence']].astype(str)
            st.sidebar.text("Saving Predictions...")
            st.sidebar.text("Completed!")
    
            st.markdown(get_download_link(df_out, 'prediction_data.csv'), unsafe_allow_html=True)
            st.write("### Predictions")
            st.write(df_out)

if __name__ == "__main__":
    main()
