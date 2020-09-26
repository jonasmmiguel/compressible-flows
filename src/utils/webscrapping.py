from requests import get

# Import BeautifulSoup from bs4 because make the html parse and help us to
# handle de DOM.
from bs4 import BeautifulSoup

# Import closing for ensure that any network resource will free when they go out
# of scope.
from contextlib import closing


def get_response(url):
    """ Return the raw_html for parsing later or None if can't reach the page

    :param url:
        The string for the GET request.

    :rtype: BeautifulSoup Object

    :rtype: None if can't reach the website

    """

    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except:
        print('Not found')
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def get_phase_change_html_table(Name: str) -> object:

    """ Return the html already parsed using the a helper function listed below.

    :param Name:
        A string with the name of the compound in English.

    :rtype: BeautifulSoup Object

    """

    # The name parameter is part of the url. For example, if you want the
    # methane data, the url is
    # https://webbook.nist.gov/cgi/cbook.cgi?Name=methane&Mask=4.
    url = str.format('https://webbook.nist.gov/cgi/cbook.cgi?Name={0}&Units=SI&Mask=4', Name.upper())

    # Function to get the request made, see below.
    raw_html = get_response(url)

    # Parse the html using BeautifulSoup.
    html = BeautifulSoup(raw_html, 'html.parser')

    # Extract the table that contains the data, the table has a specific
    # attributes 'aria-label' as 'Antoine Equation Parameters'.
    table = html.find('table', attrs={'aria-label': 'One dimensional data'})

    return table


def get_row_props(row: object) -> dict:
    cols = row.find_all('td')

    qty = cols[0].text
    content = cols[1].text
    unit = cols[2].text
    reference = cols[4].text

    if '±' in content:
        value = float(content.replace(' ', '').split('±')[0])
        sigma = float(content.replace(' ', '').split('±')[1])
    else:
        value = float(content)
        sigma = None

    row_data = {
        'value': value,
        'sigma': sigma,
        'unit': unit,
        'reference': reference,
    }
    return qty, row_data


def get_phase_change_data(Name: str) -> dict:
    table = get_phase_change_html_table(Name)

    rows_singlepoint = table.find_all('tr', class_='cal')
    rows_multipoints = table.find_all('tr', class_='exp')

    props = {}
    for row in rows_singlepoint + rows_multipoints:
        qty, row_data = get_row_props(row)

        # prevent older datapoints (lower rows) from replacing the most recent datapoint (upper row)
        if qty not in props.keys():
            props[qty] = row_data

    return props


def get_crit_state(name: str, ureg: object) -> list:
    phase_change_data = get_phase_change_data(name)

    Tc = phase_change_data['Tc']['value'] * ureg(phase_change_data['Tc']['unit'])
    pc = phase_change_data['Pc']['value'] * ureg(phase_change_data['Pc']['unit'])

    return Tc, pc
