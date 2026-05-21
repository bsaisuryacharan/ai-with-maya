from typing import List, Dict, Any
from pathlib import Path

class DocumentLoader:
    """Load documents from various formats."""
    
    @staticmethod
    def load_text_file(path: str) -> Dict[str, Any]:
        """Load a plain text file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {
            "content": content,
            "source": Path(path).name,
            "type": "text"
        }
    
    @staticmethod
    def load_text_files(directory: str) -> List[Dict[str, Any]]:
        """Load all .txt files from a directory."""
        docs = []
        for file_path in Path(directory).glob("*.txt"):
            docs.append(DocumentLoader.load_text_file(str(file_path)))
        return docs