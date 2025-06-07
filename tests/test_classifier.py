import unittest
from backend.services.classifier import TextClassifier  # Adjust import to your actual classifier class

class TestTextClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TextClassifier()

    def test_classify_simple_text(self):
        result = self.classifier.classify("This is a sample legal document.")
        self.assertIsInstance(result, str)  # Assuming classify returns a string label

    def test_classify_empty_text(self):
        result = self.classifier.classify("")
        self.assertIsInstance(result, str)

if __name__ == "__main__":
    unittest.main()
