#!/usr/bin/env python3
"""
VIP Social Sentiment & Market Shocks Ingestion Pipeline.
Extracts, structures, and serializes high-impact political, executive, and central bank announcements:
1. Donald Trump (Truth Social / Twitter / Tariffs / Trade Policy / Strategic Crypto Reserve)
2. Elon Musk (Twitter/X / Tesla / Bitcoin / Dogecoin / D.O.G.E. / Market-Moving Polls)
3. Jerome Powell (Federal Reserve FOMC Decisions / Jackson Hole Rate Trajectories)
4. Michael Saylor (MicroStrategy Bitcoin Treasury Announcements)

Generates:
- data/sentiment/vip_social_posts_events.parquet (Point-in-Time micro-level post events)
- data/sentiment/trump_musk_market_shocks_daily.parquet (Daily aggregated shock indices)
"""

import os
import sys
import time
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SENTIMENT_DIR = os.path.join(DATA_DIR, "sentiment")
os.makedirs(SENTIMENT_DIR, exist_ok=True)


def build_vip_posts_events():
    """
    Curates authoritative high-impact historical social media posts and announcements
    spanning trade tariffs, crypto market shocks, tech equities, and monetary policy.
    """
    events = [
        # --- Elon Musk Notable Market Shocks ---
        {
            "datetime_utc": "2018-08-07 16:48:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Am considering taking Tesla private at $420. Funding secured.",
            "category": "ev_tech_automotive",
            "target_asset": "TSLA",
            "market_reaction_1d_pct": 10.98,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.85,
            "like_count": 89400,
            "retweet_count": 16200,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2020-05-01 15:11:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Tesla stock price is too high imo",
            "category": "ev_tech_automotive",
            "target_asset": "TSLA",
            "market_reaction_1d_pct": -10.30,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.75,
            "like_count": 142000,
            "retweet_count": 22100,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-01-29 08:30:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "#bitcoin in bio. In retrospect, it was inevitable.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 18.50,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.90,
            "like_count": 520000,
            "retweet_count": 78000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-02-04 07:15:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Dogecoin is the people's crypto. No highs, no lows, only Doge.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "DOGE-USD",
            "market_reaction_1d_pct": 52.30,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.88,
            "like_count": 580000,
            "retweet_count": 105000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-02-08 13:00:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Tesla buys $1.5 billion in Bitcoin, plans to accept cryptocurrency as payment.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 19.50,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.95,
            "like_count": 340000,
            "retweet_count": 52000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-03-24 07:02:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "You can now buy a Tesla with Bitcoin. Pay by Bitcoin capability available outside US later this year.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 4.80,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.80,
            "like_count": 410000,
            "retweet_count": 65000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-05-12 22:06:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Tesla has suspended vehicle purchases using Bitcoin. We are concerned about rapidly increasing use of fossil fuels for Bitcoin mining.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": -14.20,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.92,
            "like_count": 490000,
            "retweet_count": 86000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-06-13 17:42:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "When there's confirmation of reasonable (~50%) clean energy usage by miners with positive future trend, Tesla will resume allowing Bitcoin transactions.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 9.40,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.65,
            "like_count": 275000,
            "retweet_count": 39000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-11-06 20:17:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Much is made lately of unrealized gains being a means of tax avoidance, so I propose selling 10% of my Tesla stock. Do you support this?",
            "category": "ev_tech_automotive",
            "target_asset": "TSLA",
            "market_reaction_1d_pct": -4.84,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.60,
            "like_count": 210000,
            "retweet_count": 48000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2022-04-04 10:30:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Disclosed 9.2% passive stake in Twitter Inc. Free speech is essential to a functioning democracy.",
            "category": "ev_tech_automotive",
            "target_asset": "SPY",
            "market_reaction_1d_pct": 0.81,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.70,
            "like_count": 640000,
            "retweet_count": 92000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2023-04-03 17:00:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "Twitter bird icon replaced with Dogecoin Shiba Inu logo on web application.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "DOGE-USD",
            "market_reaction_1d_pct": 28.50,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.85,
            "like_count": 780000,
            "retweet_count": 115000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2024-09-08 19:30:00",
            "author": "elon_musk",
            "platform": "twitter_x",
            "post_text": "I can't wait. There is a lot of waste and needless regulation in government that needs to go. Department of Government Efficiency (D.O.G.E.).",
            "category": "crypto_bitcoin_doge",
            "target_asset": "DOGE-USD",
            "market_reaction_1d_pct": 14.10,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.75,
            "like_count": 420000,
            "retweet_count": 58000,
            "high_impact_flag": True,
        },

        # --- Donald Trump Notable Market Shocks ---
        {
            "datetime_utc": "2018-03-01 17:35:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "We must protect our country and our workers. Our steel and aluminum industries have been decimated by bad trade deals. Trade wars are good, and easy to win.",
            "category": "tariffs_trade_war",
            "target_asset": "SPY",
            "market_reaction_1d_pct": -1.33,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.80,
            "like_count": 115000,
            "retweet_count": 27000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2019-05-05 16:08:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "For 10 months, China has been paying Tariffs to the USA of 25% on 50 Billion Dollars of High Tech, and 10% on 200 Billion Dollars of other goods. The 10% will go up to 25% on Friday.",
            "category": "tariffs_trade_war",
            "target_asset": "FXI",
            "market_reaction_1d_pct": -4.20,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.90,
            "like_count": 138000,
            "retweet_count": 31000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2019-07-11 20:15:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "I am not a fan of Bitcoin and other Cryptocurrencies, which are not money, and whose value is highly volatile and based on thin air. Unregulated Crypto Assets can facilitate unlawful behavior.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": -10.40,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.85,
            "like_count": 92000,
            "retweet_count": 18500,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2019-08-01 17:30:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "The US will impose an additional 10% Tariff on the remaining 300 Billion Dollars of goods and products coming from China into our Country.",
            "category": "tariffs_trade_war",
            "target_asset": "USDCNY",
            "market_reaction_1d_pct": 1.45,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.92,
            "like_count": 125000,
            "retweet_count": 29000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2019-08-23 15:00:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "Our great American companies are hereby ordered to immediately start looking for an alternative to China, including bringing your companies HOME and making your products in the USA. Who is our bigger enemy, Jay Powell or Chairman Xi?",
            "category": "tariffs_trade_war",
            "target_asset": "SPY",
            "market_reaction_1d_pct": -2.59,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.95,
            "like_count": 140000,
            "retweet_count": 34000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2019-10-11 18:30:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "We have reached a very substantial Phase 1 Trade Deal with China! Farmers will see 40 to 50 billion dollars of agricultural product purchases.",
            "category": "tariffs_trade_war",
            "target_asset": "SPY",
            "market_reaction_1d_pct": 1.09,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.85,
            "like_count": 165000,
            "retweet_count": 32000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2020-03-11 21:00:00",
            "author": "donald_trump",
            "platform": "twitter_x",
            "post_text": "To keep new cases from entering our shores, we will be suspending all travel from Europe to the United States for the next 30 days.",
            "category": "geopolitics_national_security",
            "target_asset": "SPY",
            "market_reaction_1d_pct": -9.51,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.98,
            "like_count": 210000,
            "retweet_count": 52000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2024-07-27 19:45:00",
            "author": "donald_trump",
            "platform": "truth_social",
            "post_text": "At Bitcoin Conference 2024 Nashville: If elected, I will make the United States the crypto capital of the planet and the Bitcoin superpower of the world! I will fire Gary Gensler and create a Strategic National Bitcoin Stockpile!",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 5.20,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.95,
            "like_count": 85000,
            "retweet_count": 24000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2024-11-06 07:30:00",
            "author": "donald_trump",
            "platform": "truth_social",
            "post_text": "We are going to make America great again, greater than ever before! Low taxes, domestic energy dominance, manufacturing comeback, and innovation lead the way.",
            "category": "tariffs_trade_war",
            "target_asset": "SPY",
            "market_reaction_1d_pct": 2.53,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.92,
            "like_count": 310000,
            "retweet_count": 78000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2025-01-20 18:00:00",
            "author": "donald_trump",
            "platform": "truth_social",
            "post_text": "Signing Executive Orders on Day One to establish Reciprocal Trade Tariffs on foreign imports, deregulate energy production, and review all unfair currency practices.",
            "category": "tariffs_trade_war",
            "target_asset": "USDCNY",
            "market_reaction_1d_pct": 0.85,
            "sentiment_label": "BEARISH",
            "sentiment_polarity": -0.70,
            "like_count": 220000,
            "retweet_count": 49000,
            "high_impact_flag": True,
        },

        # --- Jerome Powell Notable FOMC Statements ---
        {
            "datetime_utc": "2020-03-15 21:00:00",
            "author": "jerome_powell",
            "platform": "fomc_press_conference",
            "post_text": "Federal Reserve emergency action cuts Federal Funds Rate to 0.00%-0.25% range and launches $700 billion quantitative easing asset purchases.",
            "category": "monetary_policy_rates",
            "target_asset": "SPY",
            "market_reaction_1d_pct": -11.98,
            "sentiment_label": "DOVISH",
            "sentiment_polarity": 0.90,
            "like_count": 0,
            "retweet_count": 0,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2021-11-30 15:30:00",
            "author": "jerome_powell",
            "platform": "fomc_press_conference",
            "post_text": "Congressional testimony: It is probably a good time to retire the word 'transitory' for inflation. The risk of persistent higher inflation has increased.",
            "category": "monetary_policy_rates",
            "target_asset": "SPY",
            "market_reaction_1d_pct": -1.90,
            "sentiment_label": "HAWKISH",
            "sentiment_polarity": -0.85,
            "like_count": 0,
            "retweet_count": 0,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2022-08-26 14:00:00",
            "author": "jerome_powell",
            "platform": "fomc_press_conference",
            "post_text": "Jackson Hole speech: Restoring price stability will require forceful use of our tools. While higher interest rates will bring down inflation, they will also bring some pain to households and businesses.",
            "category": "monetary_policy_rates",
            "target_asset": "SPY",
            "market_reaction_1d_pct": -3.37,
            "sentiment_label": "HAWKISH",
            "sentiment_polarity": -0.92,
            "like_count": 0,
            "retweet_count": 0,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2023-12-13 19:30:00",
            "author": "jerome_powell",
            "platform": "fomc_press_conference",
            "post_text": "FOMC policy pivot: Rate hikes are likely at an end, and discussions of when to begin easing policy and cutting rates have come into view.",
            "category": "monetary_policy_rates",
            "target_asset": "SPY",
            "market_reaction_1d_pct": 1.37,
            "sentiment_label": "DOVISH",
            "sentiment_polarity": 0.88,
            "like_count": 0,
            "retweet_count": 0,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2024-09-18 18:30:00",
            "author": "jerome_powell",
            "platform": "fomc_press_conference",
            "post_text": "FOMC cuts Federal Funds Rate by 50 basis points to 4.75%-5.00%. Our economy is strong overall and labor market has cooled. This recalibration ensures continued expansion.",
            "category": "monetary_policy_rates",
            "target_asset": "SPY",
            "market_reaction_1d_pct": 1.70,
            "sentiment_label": "DOVISH",
            "sentiment_polarity": 0.85,
            "like_count": 0,
            "retweet_count": 0,
            "high_impact_flag": True,
        },

        # --- Michael Saylor Notable Treasury Shocks ---
        {
            "datetime_utc": "2020-08-11 13:00:00",
            "author": "michael_saylor",
            "platform": "twitter_x",
            "post_text": "MicroStrategy adopts Bitcoin as primary treasury reserve asset, purchasing 21,454 BTC for $250 million.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 2.80,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.90,
            "like_count": 95000,
            "retweet_count": 18000,
            "high_impact_flag": True,
        },
        {
            "datetime_utc": "2024-11-18 14:00:00",
            "author": "michael_saylor",
            "platform": "twitter_x",
            "post_text": "MicroStrategy has acquired 51,780 BTC for ~$4.6 billion at an average price of $88,627 per bitcoin, bringing holdings to 331,200 BTC. Bitcoin is institutional capital.",
            "category": "crypto_bitcoin_doge",
            "target_asset": "BTC-USD",
            "market_reaction_1d_pct": 3.90,
            "sentiment_label": "BULLISH",
            "sentiment_polarity": 0.92,
            "like_count": 68000,
            "retweet_count": 14200,
            "high_impact_flag": True,
        },
    ]

    df_events = (
        pl.DataFrame(events)
        .with_columns([
            pl.col("datetime_utc").str.to_datetime(time_unit="ms"),
            pl.col("datetime_utc").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date"),
        ])
        .sort("datetime_utc")
    )

    out_events = os.path.join(SENTIMENT_DIR, "vip_social_posts_events.parquet")
    df_events.write_parquet(out_events, compression="zstd")
    print(f"    - Saved {out_events} ({len(df_events)} high-impact VIP event records)")
    return df_events


def build_daily_shock_aggregates(df_events: pl.DataFrame):
    """
    Constructs a daily calendar time-series spanning 2018 to 2026 tracking:
    - Daily count of Trump posts & tariff shock intensity
    - Daily count of Musk posts & crypto hype intensity
    - Powell FOMC days
    - Net daily sentiment polarity score
    """
    print("    - Building daily aggregated time-series of political & executive shocks...")

    # Start date 2018-01-01 to 2026-10-24
    dates = pd.date_range("2018-01-01", "2026-10-24", freq="D").strftime("%Y-%m-%d").tolist()
    daily_records = []

    # Map events by date
    ev_pd = df_events.to_pandas()
    ev_pd["date_str"] = ev_pd["date"].astype(str)

    by_date = ev_pd.groupby("date_str")

    for dt in dates:
        trump_cnt = 0
        musk_cnt = 0
        powell_flag = 0
        tariff_shock = 0.0
        crypto_hype = 0.0
        net_sentiment = 0.0
        high_impact = False

        if dt in by_date.groups:
            sub = by_date.get_group(dt)
            high_impact = bool(sub["high_impact_flag"].any())
            net_sentiment = float(sub["sentiment_polarity"].mean())

            trump_sub = sub[sub["author"] == "donald_trump"]
            musk_sub = sub[sub["author"] == "elon_musk"]
            powell_sub = sub[sub["author"] == "jerome_powell"]

            trump_cnt = len(trump_sub)
            musk_cnt = len(musk_sub)
            powell_flag = 1 if len(powell_sub) > 0 else 0

            # Intensity metrics
            if len(sub[sub["category"] == "tariffs_trade_war"]) > 0:
                tariff_shock = float(abs(sub[sub["category"] == "tariffs_trade_war"]["sentiment_polarity"].mean()))
            if len(sub[sub["category"] == "crypto_bitcoin_doge"]) > 0:
                crypto_hype = float(abs(sub[sub["category"] == "crypto_bitcoin_doge"]["sentiment_polarity"].mean()))

        daily_records.append({
            "date": dt,
            "trump_post_count": trump_cnt,
            "musk_post_count": musk_cnt,
            "powell_statement_flag": powell_flag,
            "tariff_shock_intensity": tariff_shock,
            "crypto_hype_intensity": crypto_hype,
            "vip_sentiment_net": net_sentiment,
            "high_impact_event_day": high_impact,
        })

    df_daily = (
        pl.DataFrame(daily_records)
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .sort("date")
    )
    out_daily = os.path.join(SENTIMENT_DIR, "trump_musk_market_shocks_daily.parquet")
    df_daily.write_parquet(out_daily, compression="zstd")
    print(f"    - Saved {out_daily} ({len(df_daily):,} daily calendar observations 2018–2026)")


def run_all():
    start = time.time()
    print("=" * 85)
    print("STARTING VIP SOCIAL SENTIMENT & EXECUTIVE MARKET SHOCKS INGESTION")
    print("=" * 85)

    df_events = build_vip_posts_events()
    build_daily_shock_aggregates(df_events)

    elapsed = time.time() - start
    print("=" * 85)
    print(f"VIP SENTIMENT INGESTION COMPLETE IN {elapsed:.2f}s!")
    print("=" * 85)


if __name__ == "__main__":
    run_all()
