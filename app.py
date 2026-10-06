
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="ShareSense ML Trading Game",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# SETTINGS
# ============================================================
STARTING_CASH = 100000.0
RANDOM_SEED = 42

STOCKS = {
    "MVA Motors": {"symbol": "MVAM", "start": 520},
    "Bharat Tech": {"symbol": "BHTC", "start": 840},
    "GreenPower": {"symbol": "GRNP", "start": 310},
    "FoodBasket": {"symbol": "FDBS", "start": 460},
    "Nova Bank": {"symbol": "NOVB", "start": 690},
}

# These are fictional companies for a safe paper-trading game.
# Prices are simulated and are NOT real market prices.

# ============================================================
# MARKET DATA GENERATOR
# ============================================================
@st.cache_data
def create_market_data():
    all_data = {}

    for i, (name, info) in enumerate(STOCKS.items()):
        rng = np.random.default_rng(RANDOM_SEED + i)

        n = 280
        dates = pd.date_range("2025-01-01", periods=n, freq="B")

        # Slowly changing hidden trend + random market noise.
        hidden_trend = (
            0.0004
            + 0.0012 * np.sin(np.arange(n) / 18)
            + 0.0008 * np.sin(np.arange(n) / 42)
        )

        returns = hidden_trend + rng.normal(0, 0.012, n)

        # A few market shocks.
        shock_days = rng.choice(np.arange(20, n - 20), size=7, replace=False)
        returns[shock_days] += rng.normal(0, 0.045, len(shock_days))

        close = np.empty(n)
        close[0] = info["start"]

        for j in range(1, n):
            close[j] = close[j - 1] * (1 + returns[j])

        close = np.maximum(close, 30)

        high = close * (1 + rng.uniform(0.002, 0.025, n))
        low = close * (1 - rng.uniform(0.002, 0.025, n))
        open_price = close * (1 + rng.normal(0, 0.006, n))
        volume = rng.integers(50000, 500000, n)

        df = pd.DataFrame({
            "Date": dates,
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })

        # ML features
        df["Return_1"] = df["Close"].pct_change()
        df["Return_3"] = df["Close"].pct_change(3)
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_Ratio"] = df["MA_5"] / df["MA_10"] - 1
        df["Volatility_10"] = df["Return_1"].rolling(10).std()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # Target = next trading day's direction.
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        all_data[name] = df

    return all_data


# ============================================================
# ML MODEL
# ============================================================
FEATURES = [
    "Return_1",
    "Return_3",
    "MA_Ratio",
    "Volatility_10",
    "RSI_14",
]

def train_model(df, current_day):
    # Train only on information available BEFORE the current game day.
    train_df = df.iloc[:current_day].dropna().copy()

    if len(train_df) < 60:
        return None, 0.0

    split = int(len(train_df) * 0.80)
    if split < 40 or split >= len(train_df):
        return None, 0.0

    x_train = train_df[FEATURES].iloc[:split]
    y_train = train_df["Target"].iloc[:split]

    x_test = train_df[FEATURES].iloc[split:]
    y_test = train_df["Target"].iloc[split:]

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        random_state=RANDOM_SEED,
        min_samples_leaf=3,
        class_weight="balanced"
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    return model, accuracy


def get_prediction(df, current_day):
    model, accuracy = train_model(df, current_day)

    current_row = df.iloc[[current_day]].dropna()

    if model is None or current_row.empty:
        return "Not enough data", 0.50, accuracy, None

    proba = model.predict_proba(current_row[FEATURES])[0]

    # Handle models that contain only one class.
    if len(model.classes_) == 1:
        up_probability = 1.0 if model.classes_[0] == 1 else 0.0
    else:
        class_to_prob = dict(zip(model.classes_, proba))
        up_probability = float(class_to_prob.get(1, 0.0))

    prediction = "UP 📈" if up_probability >= 0.50 else "DOWN 📉"
    confidence = max(up_probability, 1 - up_probability)

    return prediction, confidence, accuracy, model


# ============================================================
# GAME STATE
# ============================================================
def reset_game():
    st.session_state.cash = STARTING_CASH
    st.session_state.holdings = {name: 0 for name in STOCKS}
    st.session_state.trade_history = []
    st.session_state.day_index = 180
    st.session_state.total_trades = 0
    st.session_state.correct_predictions = 0


if "cash" not in st.session_state:
    reset_game()

market = create_market_data()


# ============================================================
# PORTFOLIO CALCULATIONS
# ============================================================
def current_prices(day_index):
    return {
        name: float(df.iloc[day_index]["Close"])
        for name, df in market.items()
    }


def portfolio_value(day_index):
    prices = current_prices(day_index)
    stock_value = sum(
        st.session_state.holdings[name] * prices[name]
        for name in STOCKS
    )
    return st.session_state.cash + stock_value


def get_total_return(day_index):
    value = portfolio_value(day_index)
    return (value / STARTING_CASH - 1) * 100


# ============================================================
# TITLE
# ============================================================
st.title("📈 ShareSense — ML Share Market Prediction Game")
st.write(
    "A classroom-style paper-trading game using Data Science + Machine Learning. "
    "All companies and prices are fictional; no real money is involved."
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🎮 Game Controls")

    selected_stock = st.selectbox("Choose a company", list(STOCKS.keys()))

    action = st.radio("Action", ["BUY", "SELL"])

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=500,
        value=1,
        step=1
    )

    if st.button("✅ Execute Trade", use_container_width=True):
        day = st.session_state.day_index
        price = float(market[selected_stock].iloc[day]["Close"])
        total_cost = quantity * price

        if action == "BUY":
            if total_cost <= st.session_state.cash:
                st.session_state.cash -= total_cost
                st.session_state.holdings[selected_stock] += quantity

                st.session_state.trade_history.append({
                    "Date": str(market[selected_stock].iloc[day]["Date"].date()),
                    "Company": selected_stock,
                    "Action": "BUY",
                    "Qty": quantity,
                    "Price": round(price, 2),
                    "Value": round(total_cost, 2)
                })
                st.session_state.total_trades += 1
                st.success("Virtual BUY completed.")
            else:
                st.error("Not enough virtual cash.")

        else:
            if quantity <= st.session_state.holdings[selected_stock]:
                st.session_state.cash += total_cost
                st.session_state.holdings[selected_stock] -= quantity

                st.session_state.trade_history.append({
                    "Date": str(market[selected_stock].iloc[day]["Date"].date()),
                    "Company": selected_stock,
                    "Action": "SELL",
                    "Qty": quantity,
                    "Price": round(price, 2),
                    "Value": round(total_cost, 2)
                })
                st.session_state.total_trades += 1
                st.success("Virtual SELL completed.")
            else:
                st.error("You do not own enough virtual shares.")

    st.divider()

    if st.button("⏭️ Next Market Day", use_container_width=True):
        if st.session_state.day_index < 255:
            old_day = st.session_state.day_index
            new_day = old_day + 1

            # Check how the AI prediction performed for each stock.
            for name, df in market.items():
                if old_day < len(df) - 1:
                    pred, _, _, _ = get_prediction(df, old_day)
                    actual_up = df.iloc[new_day]["Close"] > df.iloc[old_day]["Close"]
                    if pred != "Not enough data":
                        predicted_up = pred.startswith("UP")
                        if predicted_up == actual_up:
                            st.session_state.correct_predictions += 1

            st.session_state.day_index = new_day
            st.rerun()
        else:
            st.warning("The demo market has reached its final day.")

    if st.button("🔄 Reset Game", use_container_width=True):
        reset_game()
        st.rerun()

# ============================================================
# MAIN DATA
# ============================================================
day = st.session_state.day_index
prices = current_prices(day)

cash = st.session_state.cash
total_value = portfolio_value(day)
profit = total_value - STARTING_CASH
return_pct = get_total_return(day)

# ============================================================
# KPI CARDS
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("💰 Cash", f"₹{cash:,.0f}")
c2.metric("📦 Portfolio", f"₹{total_value:,.0f}")
c3.metric("🏆 P/L", f"₹{profit:,.0f}", f"{return_pct:.2f}%")
c4.metric("📅 Game Day", f"{day - 179}/76")
c5.metric("🔢 Trades", st.session_state.total_trades)

st.divider()

# ============================================================
# PREDICTION PANEL
# ============================================================
df = market[selected_stock]
prediction, confidence, model_accuracy, model = get_prediction(df, day)

current_price = float(df.iloc[day]["Close"])
previous_price = float(df.iloc[day - 1]["Close"])

change_pct = (current_price / previous_price - 1) * 100

left, right = st.columns([1.15, 1])

with left:
    st.subheader(f"🤖 AI Prediction — {selected_stock}")
    st.metric(
        "Current simulated price",
        f"₹{current_price:,.2f}",
        f"{change_pct:+.2f}%"
    )

    st.info(f"Prediction: **{prediction}**")
    st.progress(int(confidence * 100))
    st.write(f"AI confidence: **{confidence * 100:.1f}%**")
    st.write(f"Model test accuracy: **{model_accuracy * 100:.1f}%**")

    if model is not None:
        st.caption(
            "The Random Forest model uses recent returns, moving-average ratio, "
            "volatility, and RSI-like momentum features."
        )

with right:
    st.subheader("🧠 How to Play")
    st.write(
        "1. Read the AI prediction.\n"
        "2. Decide whether to BUY, SELL, or do nothing.\n"
        "3. Move to the next market day.\n"
        "4. Try to grow your virtual portfolio.\n"
        "5. Compare your result with the AI prediction accuracy."
    )

    st.warning(
        "Educational simulation only. This is not financial advice and "
        "the generated prices are not real market data."
    )

# ============================================================
# PRICE CHART
# ============================================================
st.subheader(f"📊 {selected_stock} Price History")

chart_start = max(0, day - 60)
chart_df = df.iloc[chart_start:day + 1].copy()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["Close"],
    mode="lines",
    name="Price"
))

fig.add_trace(go.Scatter(
    x=chart_df["Date"],
    y=chart_df["MA_10"],
    mode="lines",
    name="10-Day MA"
))

fig.update_layout(
    height=430,
    xaxis_title="Date",
    yaxis_title="Simulated Price (₹)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# HOLDINGS TABLE
# ============================================================
st.subheader("💼 Your Holdings")

rows = []
for name in STOCKS:
    qty = st.session_state.holdings[name]
    price = prices[name]
    value = qty * price

    rows.append({
        "Company": name,
        "Symbol": STOCKS[name]["symbol"],
        "Shares": qty,
        "Current Price": round(price, 2),
        "Market Value": round(value, 2),
    })

holdings_df = pd.DataFrame(rows)
st.dataframe(holdings_df, use_container_width=True, hide_index=True)

# ============================================================
# PERFORMANCE
# ============================================================
st.subheader("🏆 Game Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Portfolio Return", f"{return_pct:.2f}%")
    st.metric(
        "AI Correct Predictions",
        st.session_state.correct_predictions
    )

with col2:
    ai_total = max(1, st.session_state.day_index - 180 + 1)
    ai_accuracy_so_far = (
        st.session_state.correct_predictions / ai_total
    ) * 100
    st.metric(
        "AI Direction Accuracy So Far",
        f"{ai_accuracy_so_far:.1f}%"
    )

# ============================================================
# TRADE HISTORY
# ============================================================
st.subheader("🧾 Trade History")

if st.session_state.trade_history:
    history_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(
        history_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No trades yet. Use the sidebar to place your first virtual trade.")

# ============================================================
# LEADERBOARD
# ============================================================
st.subheader("🥇 Demo Leaderboard")

player_score = total_value
leaderboard = pd.DataFrame({
    "Player": ["AI Bot", "You", "Trend Master", "Lucky Trader"],
    "Virtual Portfolio": [
        STARTING_CASH * 1.08,
        player_score,
        STARTING_CASH * 1.05,
        STARTING_CASH * 0.97
    ]
}).sort_values("Virtual Portfolio", ascending=False).reset_index(drop=True)

leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
leaderboard["Virtual Portfolio"] = leaderboard["Virtual Portfolio"].map(
    lambda x: f"₹{x:,.0f}"
)
st.dataframe(leaderboard, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "ShareSense is a fictional educational simulation built with Python, "
    "Pandas, NumPy, scikit-learn, Streamlit, and Plotly."
)
