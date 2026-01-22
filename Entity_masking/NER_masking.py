import spacy
from typing import List


class ContextualMasker:
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        
        self.nlp = spacy.load(model_name)
        

    def mask(self, text: str, entities_to_mask: List[str]) -> str:
      
        doc = self.nlp(text)

        masked_parts = []
        last_idx = 0

        for ent in doc.ents:
            if ent.label_ in entities_to_mask:
                
                masked_parts.append(text[last_idx:ent.start_char])

                masked_parts.append(f"[{ent.label_}]")

                last_idx = ent.end_char

        masked_parts.append(text[last_idx:])

        return "".join(masked_parts)


    # to check how the entities are being recognized (#not necessary for masking)
    def analyze_entities(self, text: str):

        doc = self.nlp(text)
        data = []

        for ent in doc.ents:
            data.append(
                f"{ent.text} ({ent.label_}) - {spacy.explain(ent.label_)}"
            )

        return data


# test
if __name__ == "__main__":
    masker = ContextualMasker("en_core_web_sm")

    entities = ["ORG", "GPE", "DATE"]

    text_1 = "Apple is opening a new store in Chicago next Monday."
    print()
    print()
    print("Entities:", masker.analyze_entities(text_1))
    print("Masked 1:", masker.mask(text_1, entities))
    print()
    text_2 = "I ate an apple for lunch."
    print()
    print("Entities:", masker.analyze_entities(text_2))
    print("Masked 2:", masker.mask(text_2, entities))
