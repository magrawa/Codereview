import os
import re
import logging
from typing import List, Union, Callable, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_average(numbers: List[Union[float, int]]) -> float:
    """Calculate the average of a list of numbers.

    Args:
        numbers: A list of numbers to calculate the average from.

    Returns:
        The average of the numbers in the input list.
        Returns 0.0 if the input list is empty to avoid ZeroDivisionError.

    Raises:
        TypeError: If the input is not a list or if the list contains
            non-numeric elements.
    """
    if not isinstance(numbers, list):
        raise TypeError("Input must be a list.")

    if not numbers:
        return 0.0  # Return 0.0 for an empty list to avoid ZeroDivisionError

    for num in numbers:
        if not isinstance(num, (int, float)):
            raise TypeError("List elements must be numeric (int or float).")

    return sum(numbers) / len(numbers)


def process_data(data: list, transform_func: Optional[Callable] = None) -> list:
    """Process a data list by applying an optional transformation function
    and filtering.

    Args:
        data: A list of data elements to process.
        transform_func: A function to apply to each element. Defaults to None.

    Returns:
        A new list containing the processed data elements.  Elements
        are transformed (if transform_func is provided) and filtered
        to include only non-None values.

    Raises:
        TypeError: If transform_func is provided and is not callable.
    """
    if transform_func is not None and not callable(transform_func):
        raise TypeError("transform_func must be a callable function.")

    processed_data = []
    for item in data:
        if transform_func:
            transformed_item = transform_func(item)
        else:
            transformed_item = item

        if transformed_item is not None:
            processed_data.append(transformed_item)

    return processed_data


def read_file(filepath: str) -> List[str]:
    """Read data from a file.

    Args:
        filepath: The path to the file.

    Returns:
        A list of strings, each a line from the file.
        Returns an empty list if the file doesn't exist or an error occurs.
    """
    try:
        with open(filepath, 'r') as file:
            lines = [line.strip() for line in file]
        return lines
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        return []
    except UnicodeDecodeError as e:
        logging.error(f"UnicodeDecodeError: {e}")
        return []
    except OSError as e:
        logging.error(f"OS error occurred: {e}")
        return []
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        return []


def write_data_to_file(data: List[str], filepath: str, append: bool = False) -> None:
    """Write data to a file, appending or overwriting.

    Args:
        data: List of strings to write.
        filepath: The file path.
        append: Append if True, overwrite if False. Defaults to False.
    """
    mode = 'a' if append else 'w'

    try:
        with open(filepath, mode) as file:
            for item in data:
                file.write(str(item) + '\n')
        logging.info(f"Data written to {filepath} successfully.")
    except OSError as e:
        logging.error(f"An error occurred while writing to file: {e}")


def validate_email(email: str) -> bool:
    """Validate an email address using a regular expression.

    Args:
        email: The email address to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(email_regex, email) is not None


def sort_data(data: list, key: Optional[Callable] = None, reverse: bool = False) -> list:
    """Sort a list of data using a specified key and order.

    Args:
        data: The list of data to sort.
        key: A function to extract a comparison key from each element.
            Defaults to None.
        reverse: If True, sort in descending order. Defaults to False.

    Returns:
        A new list containing the sorted data.
    """
    return sorted(data, key=key, reverse=reverse)


def remove_duplicates(data: list) -> list:
    """Remove duplicate elements from a list while preserving order.

    Args:
        data: The list from which to remove duplicates.

    Returns:
        A new list containing only the unique elements from the input
        list, in their original order.
    """
    seen = set()
    result = []
    for item in data:
        try:
            if item not in seen:
                seen.add(item)
                result.append(item)
        except TypeError:
            logging.warning("Unhashable type encountered, skipping duplicate removal for this item.")
            result.append(item)  # Still include it, but don't check for duplicates
    return result


def group_data_by_attribute(data: List[Dict[str, Any]], attribute: str,
                              raise_key_error: bool = False) -> Dict[Any, List[Dict[str, Any]]]:
    """Group a list of dictionaries by a specified attribute.

    Args:
        data: A list of dictionaries to group.
        attribute: The key to group the dictionaries by.
        raise_key_error: If True, raise a KeyError if the attribute is not
            found. If False (default), skip dictionaries without the
            attribute, effectively excluding them from the output.

    Returns:
        A dictionary where keys are the unique values of the attribute,
        and values are lists of dictionaries that have that attribute value.
        Returns an empty dictionary if the input list is empty.

    Raises:
        KeyError: If `raise_key_error` is True and the attribute is not
            found in a dictionary.
    """
    grouped_data = {}
    if not data:
        return grouped_data

    for item in data:
        try:
            key = item[attribute]
        except KeyError:
            if raise_key_error:
                raise
            else:
                logging.debug(f"Skipping item due to missing key: {attribute}")
                continue  # Skip this item

        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(item)

    return grouped_data
