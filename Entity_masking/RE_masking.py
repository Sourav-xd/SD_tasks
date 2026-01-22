import re
from typing import Dict, Optional

class deterministicMasker:

    patterns: Dict[str, re.Pattern] = {

        'EMAIL': r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
        'PHONE_IN': r'(?<!\d)(?:\+91[\s\-]?|0)?[6-9]\d{9}(?!\d)',
        'PHONE_US': r'\b\(?\d{3}\)?[\s\.-]?\d{3}[\s\.-]?\d{4}\b',
        'PAN_ID': r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
        'CREDIT_CARD': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        
        self.compiled_patterns = {}
        for name, pattern in self.patterns.items():
            self.compiled_patterns[name] = re.compile(pattern)

        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.compiled_patterns[name] = re.compile(pattern)
    

    def mask(self, text: str) -> str:
        masked_text = text

        for entity_type, pattern in self.compiled_patterns.items():
            masked_text = pattern.sub(f"[{entity_type}]", masked_text)

        return masked_text
    

# test
if __name__ == "__main__":
    raw_text = (
        "Email me at user.name@corp.com or call +918010473297 "
        "My PAN is ABCDE1234F and card is 1234-5678-9012-3456."
    )

    masker = deterministicMasker()
    clean_text = masker.mask(raw_text)
    
    print()
    print("Original:", raw_text)
    print()
    print("Masked:  ", clean_text)
    print()