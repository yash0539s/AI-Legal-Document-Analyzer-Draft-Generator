import difflib

def compare_documents(text1: str, text2: str) -> str:
    diff = difflib.ndiff(text1.splitlines(), text2.splitlines())
    return '\n'.join([line for line in diff if line.startswith('+ ') or line.startswith('- ')])
