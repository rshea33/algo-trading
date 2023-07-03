import pytest

from src.base_model import BaseModel

# Unit test of base model to make sure that the data is loaded correctly
@pytest.mark.parametrize(
    "ticker, start, end, n_lags",
    [
        ("AAPL", "2010-01-01", "2020-01-01", 1),
        ("AAPL", "2010-01-01", "2020-01-01", 2),
        ("AAPL", "2010-01-01", "2020-01-01", 3),
        ("AAPL", "2010-01-01", "2020-01-01", 4),
    ],
)
def test_base_model(ticker, start, end, n_lags):
    """Test the base model"""
    model = BaseModel(ticker, start, end, n_lags)
    assert model.ticker == ticker
    assert model.start == start
    assert model.end == end
    assert model.n_lags == n_lags
    assert model.df.shape[0] > 0
    assert model.df.shape[1] > 0
    assert model.df.isnull().sum().sum() == 0
    assert model.df.index.is_monotonic_increasing
    assert model.df.index.is_unique

