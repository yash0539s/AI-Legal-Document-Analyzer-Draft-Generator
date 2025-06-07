import unittest
from backend.services.ner import NERService  # Adjust import

class TestNERService(unittest.TestCase):
    def setUp(self):
        self.ner = NERService()

    def test_extract_entities(self):
        text = "LexiDraft Pro is developed by OpenAI."
        entities = self.ner.extract_entities(text)
        self.assertIsInstance(entities, list)
        for ent in entities:
            self.assertIn("text", ent)
            self.assertIn("label", ent)

    def test_extract_entities_empty_text(self):
        entities = self.ner.extract_entities("")
        self.assertIsInstance(entities, list)
        self.assertEqual(len(entities), 0)

if __name__ == "__main__":
    unittest.main()
