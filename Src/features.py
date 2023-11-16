from bs4 import BeautifulSoup, Tag, NavigableString
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


paper_id = 0

wrong_citations = ['(AMT, www. mturk.com), Crowd Flower (www.crowdflower.com)', 'Upwork (www.upwork.com)' 
                   'RentACoder (www.rentacoder.com), Guru (www.guru. com)', 'NeoIT (www.neoit.com), and Smarterwork '
                                                                            '(www.smarterwork.com)',
                   'Upwork (www.upwork.com)', 'RentACoder (www.rentacoder.com), Guru (www.guru. com)',
                   "', etc.", "female,", "unmarried,", "no", "[an ITCS platform]", "no children,",
                   "bachelor's degree,", "household", "Respondent 11,", "(Economist 2004",
                   "(Economist , 2005))", "(2001)", "(2002)", "(2003)", "(2004)", "(TPI 2006)",
                   "IEEE, 1990, p. 114)", "(1978)", "(2011)", "[O, l ]",
                   "(wH ,wi,)", "L} . Thechoiceofthewageis (wn ,wL)", "[P5]"]


def extract_citations(body):
    """
    Extracts citations from the given body of text.

     param body: The body of text from which to extract citations.
     return: A list of dictionaries, each containing details about a citation.
    """
    citations = []
    for ref in body.find_all('ref', type='bibr'):
        citation_text = ref.get_text()
        citation_target = ref.get('target')

        # Remove the '#' character from the start of the citation_target
        citation_target = citation_target[1:] if citation_target and citation_target.startswith(
            '#') else citation_target

        standalone = 1 if is_standalone_citation(ref) else 0

        # Append to the citations list as a dictionary
        if citation_target and citation_text not in wrong_citations:
            citations.append({
                "citation": citation_text,
                "citation_id": citation_target,
                "standalone": standalone
            })
    return citations


def get_citation_in_quarters(body):
    """
    Extracts citation targets from the 'body' of a document, dividing them into four quarters.

    - Divides all 'div' elements in 'body' into four equal parts (quarters).
    - If there's an uneven number of 'div' elements, adds the remainder to the last quarter.
    - Finds and extracts citation references ('ref') of type 'bibr' in each quarter.
    - Strips the leading '#' from each citation target.
    - Returns four lists, each containing citation targets from a respective quarter.
    """
    # Find all div elements
    divs = body.find_all('div')

    # Calculate the number of divs for each quarter
    quarter = len(divs) // 4
    remainder = len(divs) % 4

    # Divide the divs into four arrays
    first_quarter_divs = divs[:quarter]
    second_quarter_divs = divs[quarter:2 * quarter]
    third_quarter_divs = divs[2 * quarter:3 * quarter]
    fourth_quarter_divs = divs[3 * quarter:]

    # Add the remainder to the last quarter if there are extra divs
    if remainder:
        fourth_quarter_divs.extend(divs[-remainder:])

    # Extract refs from each quarter
    first_quarter_targets = [ref['target'].lstrip('#') for div in first_quarter_divs
                             for ref in div.find_all('ref', type='bibr') if ref.has_attr('target')]
    second_quarter_targets = [ref['target'].lstrip('#') for div in second_quarter_divs
                              for ref in div.find_all('ref', type='bibr') if ref.has_attr('target')]
    third_quarter_targets = [ref['target'].lstrip('#') for div in third_quarter_divs
                             for ref in div.find_all('ref', type='bibr') if ref.has_attr('target')]
    fourth_quarter_targets = [ref['target'].lstrip('#') for div in fourth_quarter_divs
                              for ref in div.find_all('ref', type='bibr') if ref.has_attr('target')]
    return first_quarter_targets, second_quarter_targets, third_quarter_targets, fourth_quarter_targets


def is_standalone_citation(tag):
    """Determines if a citation tag is standalone based on its text and surrounding tags"""
    citation_text = tag.get_text()

    # If citation text is enclosed in square brackets, consider it as standalone
    if citation_text.startswith('[') and citation_text.endswith(']'):
        return True

    # Check a range of siblings to look for adjacent <ref> tags
    sibling = tag.previous_sibling
    while sibling and not (isinstance(sibling, NavigableString) and (')' in sibling or ']' in sibling)):
        if isinstance(sibling, Tag) and sibling.name == "ref":
            return False  # Not standalone
        sibling = sibling.previous_sibling

    sibling = tag.next_sibling
    while sibling and not (isinstance(sibling, NavigableString) and ('(' in sibling or '[' in sibling)):
        if isinstance(sibling, Tag) and sibling.name == "ref":
            return False  # Not standalone
        sibling = sibling.next_sibling

    return True  # Standalone


def compute_tf_similarity(text1, text2):
    """ Computes Term Frequency (TF) cosine similarity between two texts"""
    if not text1 or not text2:
        # If either text is empty, return None or a default value
        return None

    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    # If either vector is all zeros, return None or a default value
    if np.all(vectors[0] == 0) or np.all(vectors[1] == 0):
        return None

    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Check if the similarity is NaN and if so, return None
    if np.isnan(similarity):
        return None

    return similarity


# ------------- compute similarity using two different libraries -------------
def compute_similarity(str1, str2):
    """ Calculates similarity between two strings using SequenceMatcher """
    return SequenceMatcher(None, str1, str2).ratio()
# ----------------------------------------------


def has_author_overlap(root_author_list, ref_author_list):
    """Checks for overlap in author names between two lists, considering both full names and initials."""

    # Split names into first name and last name for root authors
    root_authors = [{'fore_name': name.split()[0], 'surname': name.split()[-1]}
                    for name in root_author_list if len(name.split()) > 1]

    # Split names into first name and last name for reference authors
    ref_authors = [{'fore_name': name.split()[0], 'surname': name.split()[-1]}
                   for name in ref_author_list if len(name.split()) > 1]

    for root_author in root_authors:
        for references_author in ref_authors:
            if (root_author['fore_name'] == references_author['fore_name']
                or root_author['fore_name'][0] == references_author['fore_name']) \
                    and root_author['surname'] == references_author['surname']:
                return 1
    return 0


def get_ref_authors(bibl_struct):
    """Extracts and combines forenames and surnames of authors from a bibliographic structure."""
    ref_authors = []
    for author in bibl_struct.find_all('author'):
        fore_name = author.find('forename').text if author.find('forename') else ""
        last_name = author.find('surname').text if author.find('surname') else ""
        ref_authors.append(fore_name + ' ' + last_name)
    return ref_authors


def get_root_authors(soup):
    """Retrieves authors' full names from the 'sourceDesc' section of XML document"""
    # Extract authors from <sourceDesc>
    root_authors = []
    for author in soup.find('sourceDesc').find_all('author'):
        fore_name = author.find('forename').text if author.find('forename') else ""
        last_name = author.find('surname').text if author.find('surname') else ""
        root_authors.append(fore_name + ' ' + last_name)
    return root_authors


def process_xml(file_path):
    """ Parses XML data from a file to extract and analyze citations, authors, and other metadata
    of each XML-file paper"""
    global paper_id
    #  Open the file and read it
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_data = file.read()
    soup = BeautifulSoup(xml_data, 'xml')
    paper_id += 1
    # Extract the title of the root paper from <titleStmt>
    title = soup.find('titleStmt').find('title').get_text()
    # Extrac the abstract of the root paper from <abstract>
    abstract = soup.find('abstract').get_text()

    root_authors = get_root_authors(soup)

    # Extract the citations from each file and check if each citation in the file standalone
    body = soup.find('body')
    citations = extract_citations(body)
    #  Extract citations per quarter
    first, second, third, fourth = get_citation_in_quarters(body)
    refs = []
    # Extract the citation_ids from the citations list
    citation_ids_only = [item['citation_id'] for item in citations]

    # Count how many times each ref_id appears in the citations list
    citation_id_counts = Counter(citation_ids_only)
    count_first = Counter(first)
    count_second = Counter(second)
    count_third = Counter(third)
    count_fourth = Counter(fourth)

    for biblStruct in soup.find('listBibl').find_all('biblStruct', recursive=False):
        ref_title = biblStruct.find('title').get_text() if biblStruct.find('title') else ""

        # Get id of each reference in the paper
        ref_id_value = biblStruct.get('xml:id')
        #  Get authors from each reference
        ref_authors = get_ref_authors(biblStruct)
        #  Detect if there is author overlap
        author_overlap = has_author_overlap(root_authors, ref_authors)
        # Detect the standalone value for this ref_id
        standalone_value = next((item['standalone'] for item in citations if item["citation_id"] == ref_id_value), 0)
        #  compute title similarity
        title_similarity = compute_similarity(title, ref_title)
        #  compute abstract and reference title similarity
        abstract_and_reference_title_similarity = compute_tf_similarity(ref_title, abstract)

        reference = {
            'root_paper_id': paper_id,
            'ref_id': ref_id_value,
            'ref_title': ref_title,
            'published': biblStruct.find('date').get_text() if biblStruct.find('date') else "",
            'ref_authors': get_ref_authors(biblStruct),
            'in_text_citation_count': citation_id_counts[ref_id_value],
            'title_similarity': title_similarity,
            'author_overlap': author_overlap,  # This will be True if there's overlap, otherwise False.
            "abstract_and_reference_title_similarity ": abstract_and_reference_title_similarity,
            'standalone': standalone_value,
            'first_quarter': count_first[ref_id_value],
            'second_quarter': count_second[ref_id_value],
            'third_quarter': count_third[ref_id_value],
            'fourth_quarter': count_fourth[ref_id_value]
        }

        refs.append(reference)
    return {
        'root_paper_title': title,
        'paper_id': paper_id,
        'root_paper_authors': root_authors,
        'in_text_citations': citations,
        'refs': refs,
    }
