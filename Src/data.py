import os
from features import process_xml
from relevant_refs import relevant_refs
import csv
path = "/Users/hishammerdan/Documents/UNI/BA_Data_Science/project-citation-classification/new_project/data/raw"
processed_csv_file_path = "/Users/hishammerdan/Documents" \
           "/UNI/BA_Data_Science/project-citation-classification/new_project/data/processed/data_model.csv"

papers = []
excluded_paper = ['Taylor2018.tei.xml', "Kokkodis2020b.tei.xml", "Du2018a.tei.xml", "Taylor2018.tei.xml",
                  "Gol2019.tei.xml"]


def label_data():
    """
    Labels each paper's references for relevance by comparing with a predefined list of relevant papers.
    - Matches each paper's title with titles in the `relevant_refs` array.
    - For each matched paper, searches its references against `relevant_refs`.
    - Labels each reference as relevant (1) if a match is found.
    """
    l_data = []

    for data in papers:
        root_paper_title = data["root_paper_title"]

        # Check if root_paper_title exists in relevant_refs
        relevant_paper = next((p for p in relevant_refs if p["paper_title"] == root_paper_title), None)

        # If relevant_paper found, let's iterate over its references and mark them in `data`
        if relevant_paper:
            for ref in data["refs"]:
                # If ref['ref_title'] is in the list of relevant_references, it's relevant, else it's not
                if ref['ref_title'] in relevant_paper["relevant_references"]:
                    ref['reference_relevance'] = 1
                else:
                    ref['reference_relevance'] = 0

                l_data.append(ref)
    return l_data


def clean_data(data_to_clean):
    """
    Cleans the data by filtering out entries without a reference title or in-text citation count.

    param data_to_clean: The data to be cleaned.
    return: A list of cleaned data.
    """
    c_data = []
    for result in data_to_clean:
        if result['ref_title'] != "" and result['in_text_citation_count']:
            c_data.append(result)

    return c_data


def save_as_csv(data):
    """
    Saves given data to a CSV file.

    - Iterates over `data`, a list of dictionaries, to extract headers from the first element.
    - Opens a new CSV file at `processed_csv_file_path`.
    - Writes headers and rows to the CSV file using `csv.DictWriter`.
    - Returns the CSV writer object.
    """
    headers = data[0].keys()
    # Create and write to the CSV file
    with open(processed_csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return writer


def main():
    """Main function to process XML files from a directory, label, clean data, and save it as a CSV file."""

    tei_dir_path = os.path.join(os.getcwd(), path)
    # List all files in the 'tei' directory
    xml_files = [f for f in os.listdir(tei_dir_path) if f.endswith('.xml')]

    # Loop through each XML file and process
    for xml_file in xml_files:
        if xml_file in excluded_paper:
            pass
        else:
            file_path = os.path.join(tei_dir_path, xml_file)
            paper_data = process_xml(file_path)
            papers.append(paper_data)

    labeled_data = label_data()
    cleaned_data = clean_data(labeled_data)
    save_as_csv(cleaned_data)


if __name__ == "__main__":
    main()
