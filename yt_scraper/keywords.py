
def get_keyword(keyword_file_path: str) -> str:
    """
    Retrieves the first keyword from a file, then rewrites the file without the first keyword.
    If the file doesn't exist or is empty, returns None.

    Args:
        keyword_file_path (str): The path to the file containing keywords.

    Returns:
        Optional[str]: The first keyword if available, None otherwise.
    """

    keyword = ''

    try:
        with open(keyword_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:  # Check if the file is empty
            return None
        
        keyword = lines[0].strip()

        # Now, rewrite the file excluding the first line
        with open(keyword_file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines[1:])  # Write from the second line onwards

        return keyword
    except FileNotFoundError:
        return None
        
    
    return keyword


