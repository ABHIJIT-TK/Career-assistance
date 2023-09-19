import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pytz


def run():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    IST_TIMEZONE = 'Asia/Kolkata'
    st.title('Stock Prediction:')
    st.write('(Tesla, Google, Microsoft, Facebook, Nvidia, Paypal, Adobe, Netflix)')

    stocks = ('TSLA', 'GOOG', 'MSFT', 'FB', 'AAPL', 'NVDA', 'PYPL', 'ADBE', 'NFLX', 'TCS', 'INFY.NS', "RELIANCE.NS",
              "TCS.NS",
              "HDFCBANK.NS",
              "HDFC.NS",
              "INFY.NS",
              "ICICIBANK.NS",
              "KOTAKBANK.NS",
              "HINDUNILVR.NS",
              "BAJFINANCE.NS",
              "SBIN.NS",
              "^NSEI"
              )
    selected_stocks = st.selectbox('Select Dataset for prediction', stocks)

    n_years = st.slider("Years of prediction:", 1, 5)
    period = n_years * 365

    @st.cache_resource
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data = data.tz_localize(pytz.utc).tz_convert(IST_TIMEZONE)  # Localize and convert to IST
        data.reset_index(inplace=True)  # Reset the index
        return data

    data_load_state = st.text('Loading Data...')
    data = load_data(selected_stocks)
    data_load_state.text('Done !!')

    st.subheader('The Data:')
    st.dataframe(data.tail())

    def plot_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_Open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_Close'))
        fig.layout.update(title_text="Time Series data with range slider", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_data()

    data_frame_train = data[['Date', 'Close']].copy()
    data_frame_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    # Remove timezone from 'ds' column
    data_frame_train['ds'] = data_frame_train['ds'].dt.tz_localize(None)

    p = Prophet()
    p.fit(data_frame_train)

    future = p.make_future_dataframe(periods=period)
    forecast = p.predict(future)

    st.subheader('Predicted output:')
    st.write(forecast.tail())

    st.subheader(f'The Forecast for {n_years} year:')
    fig1 = plot_plotly(p, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = p.plot_components(forecast)
    st.write(fig2)

if __name__ == "__main__":
    run()