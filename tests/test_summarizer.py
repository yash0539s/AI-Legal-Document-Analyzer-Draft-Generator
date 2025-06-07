import unittest
from backend.services.summarizer import Summarizer  # Adjust import

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.summarizer = Summarizer()

    def test_summarize_simple_text(self):
        text = "LexiDraft Pro is an advanced document processing system that classifies, summarizes, and extracts entities."
        summary = self.summarizer.summarize(text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    def test_summarize_empty_text(self):
        summary = self.summarizer.summarize("")
        self.assertIsInstance(summary, str)

if __name__ == "__main__":
    unittest.main()
