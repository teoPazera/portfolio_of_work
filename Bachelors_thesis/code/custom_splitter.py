import re

def sentence_split(text: str) -> list[str]:
    """
    Splits a block of text into sentences.
    This regex splits on punctuation (., !, ?) followed by whitespace or a newline.
    Args:
        text (str): text to be splitted into sentences
    Returns:
        list[str] -> list of sentences"""
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return sentences

def is_table_line(line: str) -> bool:
    """
    Determines if a line is likely part of a markdown table.
    This heuristic assumes that a table line will contain at least two pipe characters.
    Args: 
        line(str): line to be checked if it is a table line
    Returns:
        bool: True if the line is likely part of a table, False otherwise.
    """
    return line.count('|') >= 2

def split_markdown_sentences_tables(text: str) -> tuple[list[str], list[str]]:
    """
    Splits markdown text into two lists: one for sentences and one for tables.
    When a table is encountered, the last non-table line (assumed to be a header)
    is moved into the table block.
    Args:
        text (str): text to be splitted in tables and sentences

    Returns:
        tuple[list[str], list[str]] -> tuple with two lists one with sentences
          splitted individual sentences second with markdown tables
    """
    lines = text.splitlines()
    sentences = []       # List to store sentences extracted from non-table text.
    tables = []          # List to store table blocks.
    
    non_table_buffer = []  # Buffer for lines that are not part of a table.
    table_buffer = []      # Buffer for consecutive table lines.

    def flush_non_table_buffer():
        nonlocal non_table_buffer
        if non_table_buffer:
            paragraph = " ".join(non_table_buffer).strip()
            if paragraph:
                sentences.extend(sentence_split(paragraph))
            non_table_buffer = []

    def flush_table_buffer():
        nonlocal table_buffer
        if table_buffer:
            table_block = "\n".join(table_buffer).strip()
            tables.append(table_block)
            table_buffer = []

    for line in lines:
        if is_table_line(line):
            # At the start of a new table block, if available, pop the last non-table line as header.
            if not table_buffer and non_table_buffer:
                header_line = non_table_buffer.pop()
                flush_non_table_buffer()  # Flush any remaining non-table text.
                table_buffer.append(header_line)
            table_buffer.append(line)
        else:
            # If leaving a table block, flush it first.
            if table_buffer:
                flush_table_buffer()
            if line: # getting rid of empty strings
                non_table_buffer.append(line)

    # Flush any remaining text.
    flush_non_table_buffer()
    flush_table_buffer()

    return sentences, tables


def recursive_split(text: str, chunk_length: int, delimiters: list[str]=None) -> list[str]:
    """
    Recursively splits `text` into chunks of at most `chunk_length` characters.
    It uses a list of delimiters to try to split at natural boundaries.
    If none of the delimiters can reduce the chunk size below `chunk_length`,
    it will perform a hard split.
    Args:
        text (str): text to be splitted
        chunk_length (int): length of the final desired text chunks 
        delimiters (list[str]): characters to use when splitting the text, defaults to None.
    Returns:
        list[str]: list of text chunks, each at most `chunk_length` characters long.
    """
    if delimiters is None:
        # Order: try splitting on double newlines, then newline, then space.
        delimiters = ["\n#","\n##","\n###", "\n\n", "\n", " "]
        
    # Base case: if the text is short enough, return it as is.
    if len(text) <= chunk_length:
        return [text]
    
    # If no delimiters left, do a hard split.
    if not delimiters:
        return [text[i:i+chunk_length] for i in range(0, len(text), chunk_length)]
    
    delimiter = delimiters[0]
    parts = text.split(delimiter)
    # If splitting did not break the text (i.e. no occurrence of delimiter), use the next one.
    if len(parts) == 1:
        return recursive_split(text, chunk_length, delimiters[1:])
    
    chunks = []
    current_chunk = ""
    for part in parts:
        # If current_chunk is empty, candidate is just the part;
        # otherwise, add the delimiter between parts.
        candidate = part if not current_chunk else current_chunk + delimiter + part
        if len(candidate) <= chunk_length:
            current_chunk = candidate
        else:
            # current_chunk is as large as we can get it with this delimiter.
            if current_chunk:
                chunks.append(current_chunk)
            else:
                # In case a single part is larger than chunk_length,
                # we still add it (it will be further split below).
                chunks.append(part)
            current_chunk = part  # start new chunk with the current part.
    if current_chunk:
        chunks.append(current_chunk)
    
    # Recursively ensure each chunk is under the limit.
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > chunk_length:
            final_chunks.extend(recursive_split(chunk, chunk_length, delimiters[1:]))
        else:
            final_chunks.append(chunk)
    return final_chunks

def split_markdown_to_chunks(text: str, chunk_length: int) -> list[str]:
    """
    Splits the markdown text into chunks of at most `chunk_length` characters,
    while ensuring that table blocks (including a possible header immediately before)
    remain intact.
    
    The function returns a single list of strings. Non-table blocks that exceed the
    limit are recursively split using a series of delimiters.
    Args:
        text (str): markdown text to be split into chunks
        chunk_length (int): length of the final desired text chunks
    Returns: 
        list: list of text chunks, each at most `chunk_length` characters long.
    """
    lines = text.splitlines()
    chunks = []            # Final list of chunks.
    non_table_buffer = []  # Buffer for non-table text lines.
    table_buffer = []      # Buffer for consecutive table lines.

    def flush_non_table_buffer():
        nonlocal non_table_buffer
        if non_table_buffer:
            # Join non-table lines into a single block.
            paragraph = " ".join(non_table_buffer).strip()
            if paragraph:
                # Recursively split if needed.
                for part in recursive_split(paragraph, chunk_length):
                    chunks.append(part)
            non_table_buffer = []

    def flush_table_buffer():
        nonlocal table_buffer
        if table_buffer:
            # Join table lines together.
            table_block = "\n".join(table_buffer).strip()
            chunks.append("##TABLE##" + table_block)
            table_buffer.clear()

    for line in lines:
        if is_table_line(line):
            # Starting a table block:
            # If table_buffer is empty and non_table_buffer has content,
            # pop the last line (assumed header) to include with the table.
            if not table_buffer and non_table_buffer:
                header_line = non_table_buffer.pop()
                flush_non_table_buffer()  # Flush any remaining non-table text.
                table_buffer.append(header_line)
            table_buffer.append(line)
        else:
            # On encountering non-table line while in a table block, flush table.
            if table_buffer:
                flush_table_buffer()
            if line:
                non_table_buffer.append(line)

    # Flush any remaining buffers.
    flush_non_table_buffer()
    flush_table_buffer()

    return chunks


def split_large_table(table_text: str, chunk_length: int) -> list:
    """
    Splits a markdown table that is too big into smaller table chunks.
    Assumes the first non-empty table line(s) constitute the header.
    If the second line looks like a separator (e.g. contains dashes), it is also treated as header.
    Each output chunk will contain the header followed by a subset of data rows.
     Args:
        table_text (str): markdown table text to be split into table chunks
        chunk_length (int): length of the final desired text chunks

    Returns: 
        list: list of table chunks, each at most `chunk_length` characters long. with the header at the start
    """
    lines = [line for line in table_text.splitlines() if line.strip()]
    if not lines:
        return []
    
    # Assume the first line is the header.
    header = lines[0]
    # Check if second line is a separator (contains only dashes or pipes)
    if len(lines) > 1 and re.match(r'^[\s|\-:]+$', lines[1].strip()):
        header += "\n" + lines[1]
        data_rows = lines[2:]
    else:
        data_rows = lines[1:]
    
    chunks = []
    current_chunk = header  # start with header for each chunk
    for row in data_rows:
        # If adding the row would exceed the chunk limit, finish the current chunk.
        if len(current_chunk) + len("\n" + row) > chunk_length:
            chunks.append("##TABLE##" + current_chunk)
            current_chunk = header + "\n" + row
        else:
            current_chunk += "\n" + row
    if current_chunk:
        chunks.append("##TABLE##" + current_chunk)
    return chunks

if __name__ == "__main__":
    md_text = """This is a sentence. And another one!

Table Header: Some important info
| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |

This is after the table. And yet another sentence.
"""
    sent_list, table_list = split_markdown_sentences_tables(md_text)
    print(sent_list)
    print(table_list)
    # print("Sentences:")
    # for sentence in sent_list:
    #     print("-", sentence)
    
    # print("\nTables:")
    # for table in table_list:
    #     print("-----")
    #     print(table)
    #     print("-----")
