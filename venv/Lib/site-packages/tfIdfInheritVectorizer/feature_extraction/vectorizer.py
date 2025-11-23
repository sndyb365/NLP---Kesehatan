import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFVectorizer(TfidfVectorizer):
    """
    A class for performing the TF-IDF transformation on text data. This class inherits from the
    TfidfVectorizer class from scikit-learn.
    
    Parameters
    ----------
    input_data : list of str, optional
        The input text data to be transformed. The default is None.
    stop_words : list of str, optional
        A list of stop words to be ignored during the transformation. The default is None.
    *args : list
        Additional arguments to be passed to the TfidfVectorizer class.
    **kwargs : dict
        Additional keyword arguments to be passed to the TfidfVectorizer class.
    
    """
    def __init__(self, input_data=None, stop_words=None, encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', token_pattern=r'(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        """
        Initialize the TFIDFVectorizer object.
        
        Parameters
        ----------
        input_data : list of str, optional
            The input text data to be transformed. The default is None.
        stop_words : list of str, optional
            A list of stop words to be ignored during the transformation. The default is None.
        Other Parameters are taken from sklearn documentation because of the lack of *kwargs usage in skelarn inheritence
        
        """
        super().__init__(input=input_data, stop_words=stop_words,  encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,  token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        
    def fit_transform(self, input_data=None):
        """
        Fit the TF-IDF transformation to the input data and return the transformed data.
        
        Parameters
        ----------
        input_data : list of str, optional
            The input text data to be transformed. The default is None.
        
        Returns
        -------
        transformed_data : scipy sparse matrix
            The transformed data in the form of a scipy sparse matrix.
        
        """
        if input_data is not None:
            self.input_data = input_data
        return super().fit_transform(input_data)
    
    def fit(self, input_data=None):
        """
        Fit the TF-IDF transformation to the input data.
        
        Parameters
        ----------
        input_data : list of str, optional
            The input text data to be transformed. The default is None.
        
        Returns
        -------
        self : object
            The fitted TFIDFVectorizer object.
        
        """
        if input_data is not None:
            self.input_data = input_data
        super().fit(input_data)
        return self
    
    def transform(self, input_data=None):
        """
        Perform the TF-IDF transformation on the input data and return the transformed data.
        
        Parameters
        ----------
        input_data : list of str, optional
            The input text data to be transformed. The default is None.
        
        Returns
        -------
        transformed_data : scipy sparse matrix
            The transformed data in the form of a scipy sparse matrix.
        
        """
        if input_data is not None:
            self.input_data = input_data
        return super().transform(input_data)
