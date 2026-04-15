import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df = pd.read_csv("historical_data.csv")
print(" Data loaded! Total trades:", len(df))

df = df.rename(columns={
    'Timestamp': 'close_time',
    'Account': 'account_id',
    'Coin': 'symbol',
    'Side': 'trade_type',
    'Execution Price': 'entry_price',
    'Size USD': 'volume',
    'Closed PnL': 'profit'
}).copy()

df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
df = df.sort_values('close_time').reset_index(drop=True)

df['time_gap_min'] = df.groupby('account_id')['close_time'].diff().dt.total_seconds() / 60
df['time_gap_min'] = df['time_gap_min'].fillna(120)
df['date'] = df['close_time'].dt.date
df['is_win'] = df['profit'] > 0
df['is_loss'] = df['profit'] < 0

print(f"Date range: {df['close_time'].min().date()} to {df['close_time'].max().date()}\n")

# ====================== EVI CALCULATOR ======================
def calculate_evi(group, window=10):
    original_index = group.index

    if len(group) < 5:
        return pd.Series(np.nan, index=original_index)

    group = group.copy().sort_values('close_time')
    group = group.reset_index(drop=True)

    win_rate = group['is_win'].rolling(window=window, min_periods=3).mean() * 100
    avg_profit = group['profit'].rolling(window=window, min_periods=3).mean()
    profit_std = group['profit'].rolling(window=window, min_periods=3).std()
    consistency = 100 - (profit_std / (abs(avg_profit) + 1e-6) * 30)
    consistency = consistency.clip(0, 100)

    risk_control = 100 - (group['volume'].rolling(window=window, min_periods=3).mean() * 200)
    risk_control = risk_control.clip(0, 100)

    # --- Emotional analysis -----------
    loss_streak = group['is_loss'].rolling(window=window, min_periods=1).sum()
    loss_streak_score = (100 - loss_streak * 8).clip(0, 100)

    volume_shift = group['volume'].shift(-1)
    revenge = (volume_shift / (group['volume'] + 1e-6) - 1).where(group['is_loss']).fillna(0)
    revenge_score = 100 - (revenge.rolling(window=window, min_periods=3).mean() * 60).clip(0, 100)

    avg_gap = group['time_gap_min'].rolling(window=window, min_periods=3).mean()
    impulse_penalty = (60 - avg_gap).clip(0, 60) * 1.5
    impulse_score = (100 - impulse_penalty).clip(0, 100)

    symbol_change = (group['symbol'] != group['symbol'].shift(1)).astype(int)
    symbol_hop_score = (100 - symbol_change.rolling(window=window, min_periods=3).sum() * 8).clip(0, 100)

    daily_counts = group.groupby(group['close_time'].dt.date)['profit'].transform('count')
    frequency_penalty = (daily_counts - 5).clip(lower=0) * 8
    frequency_score = (100 - frequency_penalty).clip(0, 100)

    emotional_stability = (
        loss_streak_score * 0.25 +
        revenge_score     * 0.25 +
        impulse_score     * 0.25 +
        symbol_hop_score  * 0.25
    )

    evi = (
        win_rate            * 0.25 +
        consistency         * 0.20 +
        risk_control        * 0.20 +
        emotional_stability * 0.35
    )

    evi.index = original_index
    return evi


print("Calculating EVI score...")
df['EVI'] = df.groupby('account_id', group_keys=False).apply(
    lambda g: calculate_evi(g, window=10),
    include_groups=False
)
print(" EVI calculation completed!")

# ====================== AGGREGATIONS ======================

# --- Weekly ---
df['week'] = df['close_time'].dt.to_period('W').astype(str)
weekly_evi = df.groupby('week')['EVI'].mean().reset_index()
weekly_evi['week_start'] = pd.to_datetime(weekly_evi['week'].str[:10])

# --- Monthly ---
df['month'] = df['close_time'].dt.to_period('M')
monthly_evi = df.groupby('month')['EVI'].mean().reset_index()
monthly_evi['month_start'] = monthly_evi['month'].dt.to_timestamp()
monthly_evi['month_label'] = monthly_evi['month_start'].dt.strftime('%b %Y')

# --- Yearly ---
df['year'] = df['close_time'].dt.year
yearly_evi = df.groupby('year')['EVI'].mean().reset_index()
yearly_evi['year_label'] = yearly_evi['year'].astype(str)

# --- Daily (for report) ---
daily_evi = df.groupby('date')['EVI'].mean().reset_index().dropna()

print(f" Aggregations ready → Weekly: {len(weekly_evi)} | Monthly: {len(monthly_evi)} | Yearly: {len(yearly_evi)}")

# ====================== REPORT ======================
print("\n" + "="*70)
print("MY BEHAVIOR REPORT")
print("="*70)
print(f"Current Week EVI   : {weekly_evi['EVI'].iloc[-1]:.1f}/100")
print(f"Current Month EVI  : {monthly_evi['EVI'].iloc[-1]:.1f}/100")
print(f"Current Year EVI   : {yearly_evi['EVI'].iloc[-1]:.1f}/100")
print(f"Overall Average    : {daily_evi['EVI'].mean():.1f}/100")
print("="*70)

# ====================== EMOTIONAL BEHAVIOR REPORT ======================

def get_evi_label(score):
    if score >= 70:   return "Disciplined"
    elif score >= 55: return "Moderate stress"
    elif score >= 40: return "Borderline"
    else:             return "High risk"

def get_subscore_status(score):
    if score >= 70:   return "GOOD   "
    elif score >= 40: return "CAUTION"
    else:             return "HIGH RISK"

def print_emotional_report(df, weekly_evi, monthly_evi, yearly_evi, daily_evi):

    # --- Compute sub-scores from the latest 50 trades ---
    latest = df.sort_values('close_time').tail(50).copy()
    window = 10

    loss_streak       = latest['is_loss'].rolling(window, min_periods=1).sum().iloc[-1]
    loss_streak_score = max(0, min(100, 100 - loss_streak * 8))

    volume_shift  = latest['volume'].shift(-1)
    revenge       = (volume_shift / (latest['volume'] + 1e-6) - 1).where(latest['is_loss']).fillna(0)
    revenge_score = max(0, min(100, 100 - (revenge.rolling(window, min_periods=3).mean().iloc[-1] * 60)))

    avg_gap       = latest['time_gap_min'].rolling(window, min_periods=3).mean().iloc[-1]
    impulse_score = max(0, min(100, 100 - max(0, 60 - avg_gap) * 1.5))

    symbol_change    = (latest['symbol'] != latest['symbol'].shift(1)).astype(int)
    symbol_hop_score = max(0, min(100, 100 - symbol_change.rolling(window, min_periods=3).sum().iloc[-1] * 8))

    daily_counts   = latest.groupby(latest['close_time'].dt.date)['profit'].transform('count')
    freq_penalty   = max(0, daily_counts.iloc[-1] - 5) * 8
    frequency_score = max(0, min(100, 100 - freq_penalty))

    emotional_stability = (
        loss_streak_score * 0.25 +
        revenge_score     * 0.25 +
        impulse_score     * 0.25 +
        symbol_hop_score  * 0.25
    )

    week_evi  = weekly_evi['EVI'].iloc[-1]
    month_evi = monthly_evi['EVI'].iloc[-1]
    year_evi  = yearly_evi['EVI'].iloc[-1]
    overall   = daily_evi['EVI'].mean()

    
    print("\n" + "="*70)
    print("  EMOTIONAL BEHAVIOR REPORT")
    print("="*70)
    print(f"  This Week  EVI : {week_evi:5.1f}/100  →  {get_evi_label(week_evi)}")
    print(f"  This Month EVI : {month_evi:5.1f}/100  →  {get_evi_label(month_evi)}")
    print(f"  This Year  EVI : {year_evi:5.1f}/100  →  {get_evi_label(year_evi)}")
    print(f"  All-Time Avg   : {overall:5.1f}/100  →  {get_evi_label(overall)}")
    print("="*70)

    
    print("\n  EMOTION SUB-SCORES  (latest 50-trade window)")
    print(f"  {'Metric':<28} {'Score':>6}   {'Bar':^12}  Status")
    print("  " + "-"*62)

    subs = [
        ("Loss streak control",  loss_streak_score),
        ("Revenge trading",      revenge_score),
        ("Impulse control",      impulse_score),
        ("Symbol hopping",       symbol_hop_score),
        ("Trade frequency",      frequency_score),
        ("Overall emotional",    emotional_stability),
    ]
    for name, score in subs:
        filled = int(score // 10)
        bar    = "█" * filled + "░" * (10 - filled)
        status = get_subscore_status(score)
        print(f"  {name:<28} {score:5.1f}   [{bar}]  {status}")

    
    print("\n  BEHAVIORAL FLAGS")
    print("  " + "-"*62)

    flags = []

    if loss_streak_score < 40:
        flags.append(("[DANGER]", f"Loss streak spiral — {int(loss_streak)} consecutive losses. Stop and take a break."))
    elif loss_streak_score < 70:
        flags.append(("[WARN]  ", f"Loss streak building — {int(loss_streak)} losses in window. Stay disciplined."))
    else:
        flags.append(("[GOOD]  ", "Loss streak under control. Good recovery behavior."))

    if revenge_score < 40:
        flags.append(("[DANGER]", "Revenge trading confirmed — volume spikes sharply after losses."))
    elif revenge_score < 70:
        flags.append(("[WARN]  ", "Possible revenge trades — monitor your size after a loss."))
    else:
        flags.append(("[GOOD]  ", "No revenge trading detected. Size is stable after losses."))

    if impulse_score < 40:
        flags.append(("[DANGER]", f"Very impulsive — avg gap only {avg_gap:.0f} min. Slow down significantly."))
    elif impulse_score < 70:
        flags.append(("[WARN]  ", f"Some impulsiveness — avg gap {avg_gap:.0f} min (healthy target: >60 min)."))
    else:
        flags.append(("[GOOD]  ", f"Patience is strong — avg {avg_gap:.0f} min between trades."))

    if symbol_hop_score < 40:
        flags.append(("[DANGER]", "Excessive symbol hopping — focus on fewer pairs."))
    elif symbol_hop_score < 70:
        flags.append(("[WARN]  ", "Some symbol hopping detected — reduce pair switching."))
    else:
        flags.append(("[GOOD]  ", "Symbol focus is strong — disciplined pair selection."))

    if frequency_score < 40:
        flags.append(("[DANGER]", "Severe overtrading — too many trades per day."))
    elif frequency_score < 70:
        flags.append(("[WARN]  ", "High trade frequency — risk of overtrading detected."))
    else:
        flags.append(("[GOOD]  ", "Trade frequency is healthy. Not overtrading."))

    for tag, msg in flags:
        print(f"  {tag}  {msg}")

    
    print("\n  EMOTIONAL STATE VERDICT")
    print("  " + "-"*62)

    if emotional_stability >= 70:
        verdict = "You are trading with strong emotional control. Keep it up."
    elif emotional_stability >= 55:
        verdict = "Moderate emotional stress. Watch how you react after losses."
    elif emotional_stability >= 40:
        verdict = "Emotional discipline slipping. Consider reducing position size."
    else:
        verdict = "HIGH EMOTIONAL RISK. Stop trading today and review your plan."

    print(f"  Emotional EVI Score : {emotional_stability:.1f}/100")
    print(f"  Verdict             : {verdict}")
    print("="*70)


print_emotional_report(df, weekly_evi, monthly_evi, yearly_evi, daily_evi)

# ====================== SHARED CHART STYLE ======================
def style_ax(ax, title, xlabel):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(title, fontsize=13, color='white', pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color='white')
    ax.set_ylabel('EVI Score (0–100)', fontsize=11, color='white')
    ax.axhline(70, color='#3fb950', linestyle='--', lw=1.8, label='Good (70+)')
    ax.axhline(40, color='#f85149', linestyle='--', lw=1.8, label='High Risk (<40)')
    ax.set_ylim(0, 105)
    ax.legend(facecolor='#1c2128', labelcolor='white', fontsize=9)

# ====================== CHART 1 — WEEKLY ======================
fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0e1117')
ax.plot(weekly_evi['week_start'], weekly_evi['EVI'],
        color='#58a6ff', linewidth=2.5, marker='o', markersize=5,
        markeredgecolor='white', label='Weekly EVI')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
plt.xticks(rotation=45, ha='right', color='white')
style_ax(ax, 'EVI Score History — Weekly', 'Week')
plt.tight_layout()
plt.savefig('EVI_Weekly.png', dpi=150, facecolor='#0e1117')
plt.show()
print(" Weekly chart saved → EVI_Weekly.png")

# ====================== CHART 2 — MONTHLY ======================
fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0e1117')
ax.plot(monthly_evi['month_start'], monthly_evi['EVI'],
        color='#d2a8ff', linewidth=2.5, marker='s', markersize=7,
        markeredgecolor='white', label='Monthly EVI')

for _, row in monthly_evi.iterrows():
    ax.annotate(f"{row['EVI']:.1f}",
                xy=(row['month_start'], row['EVI']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=8, color='#d2a8ff')

ax.set_xticks(monthly_evi['month_start'])
ax.set_xticklabels(monthly_evi['month_label'], rotation=45, ha='right', color='white')
style_ax(ax, 'EVI Score History — Monthly', 'Month')
plt.tight_layout()
plt.savefig('EVI_Monthly.png', dpi=150, facecolor='#0e1117')
plt.show()
print("Monthly chart saved → EVI_Monthly.png")

# ====================== CHART 3 — YEARLY ======================
fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0e1117')

bars = ax.bar(yearly_evi['year_label'], yearly_evi['EVI'],
              color=['#3fb950' if v >= 70 else '#f0883e' if v >= 40 else '#f85149'
                     for v in yearly_evi['EVI']],
              edgecolor='white', linewidth=0.8, width=0.5)

for bar, val in zip(bars, yearly_evi['EVI']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}", ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold')

ax.axhline(70, color='#3fb950', linestyle='--', lw=1.8, label='Good (70+)')
ax.axhline(40, color='#f85149', linestyle='--', lw=1.8, label='High Risk (<40)')
ax.set_facecolor('#161b22')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
ax.set_title('EVI Score Summary — Yearly', fontsize=13, color='white', pad=10)
ax.set_xlabel('Year', fontsize=11, color='white')
ax.set_ylabel('Average EVI Score (0–100)', fontsize=11, color='white')
ax.set_ylim(0, 115)
ax.legend(facecolor='#1c2128', labelcolor='white', fontsize=9)
plt.xticks(color='white')
plt.tight_layout()
plt.savefig('EVI_Yearly.png', dpi=150, facecolor='#0e1117')
plt.show()
print("Yearly chart saved → EVI_Yearly.png")

# ====================== SAVE DATA ======================
df.to_csv("preprocessed_trade_logs.csv", index=False)
weekly_evi.to_csv("EVI_weekly_summary.csv", index=False)
monthly_evi.to_csv("EVI_monthly_summary.csv", index=False)
yearly_evi.to_csv("EVI_yearly_summary.csv", index=False)
print("All data saved!")
