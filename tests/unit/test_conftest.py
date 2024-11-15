import pytest
import tfs
from tests.conftest import assert_tfsdataframe_equal


class TestAssertTfsDataFrameEqual:
    @pytest.mark.basic
    def test_no_headers_equal(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert_tfsdataframe_equal(df1, df1)
    
    @pytest.mark.basic
    def test_no_headers_different_data(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = tfs.TfsDataFrame({"a": [1, 2, 2], "b": [4, 5, 6]})
        with pytest.raises(AssertionError):
            assert_tfsdataframe_equal(df1, df2)

    @pytest.mark.basic
    def test_no_headers_different_order(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = tfs.TfsDataFrame({"b": [4, 5, 6], "a": [1, 2, 3]})
        
        with pytest.raises(AssertionError):
            assert_tfsdataframe_equal(df1, df2)

        assert_tfsdataframe_equal(df1, df2, check_like=True) 
    
    @pytest.mark.basic
    def test_with_headers_equal(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        df2 = tfs.TfsDataFrame({"b": [4, 5, 6], "a": [1, 2, 3]}, headers={"a": "a", "b": "b"})
        assert_tfsdataframe_equal(df1, df1)
        
        with pytest.raises(AssertionError):
            assert_tfsdataframe_equal(df1, df2)

        assert_tfsdataframe_equal(df1, df2, check_like=True)

    @pytest.mark.basic
    def test_with_headers_different_data(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        df2 = tfs.TfsDataFrame({"a": [1, 2, 2], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        with pytest.raises(AssertionError):
            assert_tfsdataframe_equal(df1, df2)
    
    @pytest.mark.basic
    def test_with_headers_different_datatypes(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        df2 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": ['4', '5', '6']}, headers={"a": "a", "b": "b"})
        with pytest.raises(AssertionError):
            assert_tfsdataframe_equal(df1, df2)
        
        df3 = tfs.TfsDataFrame({"a": [1., 2., 3.], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        with pytest.raises(AssertionError) as e:
            assert_tfsdataframe_equal(df1, df3)
        assert "dtype" in str(e)
    
    @pytest.mark.basic
    def test_with_headers_different_headers_values(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        df2 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "c"})
        with pytest.raises(AssertionError) as e:
            assert_tfsdataframe_equal(df1, df2)
        assert "b != c" in str(e)
        
        with pytest.raises(AssertionError) as e:
            assert_tfsdataframe_equal(df1, df2, compare_keys=False)
        assert "b != c" in str(e)
    
    @pytest.mark.basic
    def test_with_headers_different_headers_keys(self):
        df1 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "b"})
        df2 = tfs.TfsDataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, headers={"a": "a", "b": "b", "c": "c"})
        with pytest.raises(AssertionError) as e:
            assert_tfsdataframe_equal(df1, df2)  # `compare_keys=True` is default
        
        # compare only common keys ---
        assert_tfsdataframe_equal(df1, df2, compare_keys=False)
            
