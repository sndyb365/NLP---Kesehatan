import unittest

from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer

class TestTFIDFVectorizer(unittest.TestCase):
    def setUp(self):
        self.text_data = [
            "This is the first document",
            "This is another sample document",
            "And this is the third one",
            "Is this the first document?"
        ]
        
    def test_transform_with_none_input(self):
        """
        Test the transform method with no input data.
        """
        vectorizer = TFIDFVectorizer()
        vectorizer.fit(self.text_data)
        with self.assertRaises(TypeError):
            vectorizer.transform()
            
    def test_fit_transform_with_none_input(self):
        """
        Test the fit_transform method with no input data.
        """
        vectorizer = TFIDFVectorizer()
        with self.assertRaises(TypeError):
            vectorizer.fit_transform()
            
    def test_fit_with_none_input(self):
        """
        Test the fit method with no input data.
        """
        vectorizer = TFIDFVectorizer()
        with self.assertRaises(TypeError):
            vectorizer.fit()
            
    def test_transform_with_incorrect_input_type(self):
        """
        Test the transform method with incorrect input data type.
        """
        vectorizer = TFIDFVectorizer(self.text_data)
        with self.assertRaises(ValueError):
            vectorizer.transform([1, 2, 3, 4])
            
    def test_fit_transform_with_incorrect_input_type(self):
        """
        Test the fit_transform method with incorrect input data type.
        """
        vectorizer = TFIDFVectorizer()
        with self.assertRaises(ValueError):
            vectorizer.fit_transform([1, 2, 3, 4])
            
    def test_fit_with_incorrect_input_type(self):
        """
        Test the fit method with incorrect input data type.
        """
        vectorizer = TFIDFVectorizer()
        with self.assertRaises(ValueError):
            vectorizer.fit([1, 2, 3, 4])
            
   
