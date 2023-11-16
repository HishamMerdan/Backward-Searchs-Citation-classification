import pandas as pd
import json
from relevant_refs import relevant_refs


filename = "data_model.csv"

def read_data(data):

    data_to_append = []

    for paper in data:
        root_paper_title = paper["root_paper_title"]

        # Check if root_paper_title exists in relevant_refs
        relevant_paper = next((p for p in relevant_refs if p["paper_title"] == root_paper_title), None)

        # If we found a relevant_paper, let's iterate over its references and mark them in `data`
        if relevant_paper:
            for ref in paper["refs"]:
                # If ref['ref_title'] is in the list of relevant_references, it's relevant, else it's not
                if ref['ref_title'] in relevant_paper["relevant_references"]:
                    ref['reference_relevance'] = 1
                else:
                    ref['reference_relevance'] = 0

                data_to_append.append(ref)
    # clean the rows with empty reference
    data = []
    for result in data_to_append:
        if result['ref_title'] != "" and result['in_text_citation_count']:
            data.append(result)
    # Convert all accumulated data to a DataFrame
    print(len(data))
    df_to_append = pd.DataFrame(data)
    try:
        # Try reading the existing CSV file
        df_existing = pd.read_csv(filename)
        # Use pandas.concat to append new data
        df_updated = pd.concat([df_existing, df_to_append], ignore_index=True)
        df_updated.to_csv(filename, index=False)
    except:
        # If there's an error (like the file doesn't exist), write a new CSV file
        df_to_append.to_csv(filename, index=False)

# Print the header of the CSV file
#print(pd.read_csv(filename)).columns.tolist)