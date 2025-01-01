import datetime
import parsedatetime
import re

def validate_date(trekking_date):
    """
    Validate and parse a date string. Handles any date format, and even relative dates.
    """
    cal = parsedatetime.Calendar()
    parsed_date, parse_status = cal.parse(trekking_date)
    
    if parse_status == 1:  # Parsing successful
        return datetime.datetime(*parsed_date[:6]).strftime('%Y-%m-%d')
    else:
        return None

def validate_email(email_str):
    """
    Validates email format using regex.
    """
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email_str) is not None