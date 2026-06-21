# %% [markdown]
# # Poisson Regression Comparison Model
#
# Side-by-side comparison: the existing correlation-weighted heuristic model
# (`02_match_level_model.py`) vs a traditional Poisson regression (GLM, fitted
# by maximum likelihood). See `03_poisson_regression_comparison_brief.md` for
# the full rationale.
#
# This script does NOT modify or re-run the production model. It reuses the
# data loading, time-weighting, and team/referee profile logic, and compares
# against the production model's already-saved predictions in
# `outputs/results/match_predictions.csv`.

# %% [markdown]
# ### Imports and Data Loading

# %%
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import poisson
from aliases import TEAM_ALIAS, REF_ALIAS

# %%
def load_match_data(file_path='data/match_level_data/match_data_combined_raw.xlsx'):
    '''
    Same loading/cleaning logic as 02_match_level_model.py, copied here so
    this script has no dependency on running 02 first.
    '''
    sheets = pd.ExcelFile(file_path).sheet_names
    all_seasons = []

    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)

        columns_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'Referee', 'HY', 'AY', 'HR', 'AR']
        if len(df.columns) > len(columns_to_keep):
            df = df[columns_to_keep].copy()

        df['home_team_id'] = df['HomeTeam'].str.strip().str.lower()
        df['away_team_id'] = df['AwayTeam'].str.strip().str.lower()
        df['home_team_id'] = df['home_team_id'].replace(TEAM_ALIAS)
        df['away_team_id'] = df['away_team_id'].replace(TEAM_ALIAS)

        df['Referee'] = df['Referee'].str.strip().str.lower()
        df['referee_id'] = (
            df['Referee'].str.split().str[0].str[0]
            + '_'
            + df['Referee'].str.split().str[-1]
        )
        df['referee_id'] = df['referee_id'].replace(REF_ALIAS)

        df['Date'] = pd.to_datetime(df['Date'])
        df['season'] = sheet
        df['match_id'] = (
            df['season'].astype(str)
            + '_' + df['Date'].dt.strftime('%Y-%m-%d')
            + '_' + df['home_team_id']
            + '_' + df['away_team_id']
        )

        df['home_cards'] = df['HY'] + df['HR']
        df['away_cards'] = df['AY'] + df['AR']
        df['total_cards'] = df['home_cards'] + df['away_cards']

        all_seasons.append(df)

    match_data = pd.concat(all_seasons, ignore_index=True)
    match_data = match_data.sort_values('Date').reset_index(drop=True)
    match_data['week'] = match_data['Date'].dt.to_period('W')
    return match_data


# %% [markdown]
# ### Time weighting and team/referee profiles
#
# Identical to 02_match_level_model.py — same decay rates, same profile
# columns. The Poisson regression needs these as raw inputs rather than as
# inputs to a correlation calculation.

# %%
def add_time_weights(history, asof_date, alpha_team=0.002, alpha_ref=0.001):
    days_ago = (asof_date - history['Date']).dt.days
    history = history.copy()
    history['weight_team'] = np.exp(-alpha_team * days_ago)
    history['weight_ref'] = np.exp(-alpha_ref * days_ago)
    return history


def wmean(x, weight):
    return (x * weight).sum() / weight.sum()


def build_home_team_profiles(history_with_weights):
    home = history_with_weights.groupby('home_team_id').apply(
        lambda t: pd.Series({
            'home_cards_for': wmean(t['home_cards'], t['weight_team']),
            'home_cards_against': wmean(t['away_cards'], t['weight_team']),
            'home_matches_count': len(t),
            'home_weights_sum': t['weight_team'].sum()
        }),
        include_groups=False
    )
    return home


def build_away_team_profiles(history_with_weights):
    away = history_with_weights.groupby('away_team_id').apply(
        lambda t: pd.Series({
            'away_cards_for': wmean(t['away_cards'], t['weight_team']),
            'away_cards_against': wmean(t['home_cards'], t['weight_team']),
            'away_matches_count': len(t),
            'away_weights_sum': t['weight_team'].sum()
        }),
        include_groups=False
    )
    return away


def build_team_profiles(history_with_weights):
    home = build_home_team_profiles(history_with_weights)
    away = build_away_team_profiles(history_with_weights)
    return home.join(away, how='outer')


def build_ref_profiles(history_with_weights):
    '''
    Unlike 02_match_level_model.py, this script actually feeds the referee
    profile into predictions (ref_cards_per_game is a regression feature).
    '''
    refs = history_with_weights.groupby('referee_id').apply(
        lambda r: pd.Series({
            'ref_cards_per_game': wmean(r['total_cards'], r['weight_ref']),
            'ref_match_count': len(r),
            'ref_weights_sum': r['weight_ref'].sum()
        }),
        include_groups=False
    )
    return refs


# %% [markdown]
# ### Building the GLM training table
#
# This is the new piece. For every match already in `history`, look up what
# the team/referee profiles looked like the week *before* that match was
# played (never using a profile informed by the match itself or anything
# after it), and assemble one row of features + actual outcome. This table
# is what the Poisson GLM is fitted on at each walk-forward step.
#
# This mirrors `fit_team_mix_weights_simple()` in 02_match_level_model.py,
# but instead of computing a correlation, it returns a feature table ready
# for `sm.GLM(..., family=sm.families.Poisson())` and adds the referee
# feature that the existing model never uses.

# %%
def build_glm_training_table(history, teams_over_time, refs_over_time, min_rows=100):
    rows = []

    for _, match in history.iterrows():
        home_team = match['home_team_id']
        away_team = match['away_team_id']
        ref_id = match['referee_id']
        match_week = match['week']

        home_profiles = teams_over_time[teams_over_time['team_id'] == home_team]
        away_profiles = teams_over_time[teams_over_time['team_id'] == away_team]

        if home_profiles.empty or away_profiles.empty:
            continue

        home_before = home_profiles[home_profiles['week'] < match_week]
        away_before = away_profiles[away_profiles['week'] < match_week]

        if home_before.empty or away_before.empty:
            continue

        home_profile = home_before.sort_values('week').iloc[-1]
        away_profile = away_before.sort_values('week').iloc[-1]

        ref_profiles = refs_over_time[refs_over_time['referee_id'] == ref_id]
        ref_before = ref_profiles[ref_profiles['week'] < match_week]
        ref_cards_per_game = (
            ref_before.sort_values('week').iloc[-1]['ref_cards_per_game']
            if not ref_before.empty else np.nan
        )

        rows.append({
            'home_for': home_profile['home_cards_for'],
            'home_against': home_profile['home_cards_against'],
            'away_for': away_profile['away_cards_for'],
            'away_against': away_profile['away_cards_against'],
            'ref_cards_per_game': ref_cards_per_game,
            'actual_home': match['home_cards'],
            'actual_away': match['away_cards'],
        })

    df = pd.DataFrame(rows)

    if len(df) < min_rows:
        return df

    # A newly promoted team can have a profile snapshot before it has ever
    # played on one side (e.g. two away matches before its first home match).
    # Those rows have no usable home/away features yet, so drop them rather
    # than feeding NaNs into the GLM fit.
    df = df.dropna(subset=['home_for', 'home_against', 'away_for', 'away_against'])

    if len(df) < min_rows:
        return df

    # Referees with no prior history yet (first appearance in the dataset):
    # fill with the training table's own mean ref rate rather than dropping
    # the match.
    df['ref_cards_per_game'] = df['ref_cards_per_game'].fillna(df['ref_cards_per_game'].mean())

    return df


# %% [markdown]
# ### Fitting the Poisson GLMs
#
# Two models, same split as the existing model's lambda calculation:
# - Home cards ~ home_for + away_against + ref_cards_per_game
# - Away cards ~ away_for + home_against + ref_cards_per_game
#
# `log(lambda) = Î²0 + Î²1*X1 + ...` â€” the log link is what makes this a
# genuine Poisson *regression* rather than a heuristic average scored with
# a Poisson distribution afterward.

# %%
HOME_FEATURES = ['home_for', 'away_against', 'ref_cards_per_game']
AWAY_FEATURES = ['away_for', 'home_against', 'ref_cards_per_game']


def fit_poisson_models(training_df):
    X_home = sm.add_constant(training_df[HOME_FEATURES])
    X_away = sm.add_constant(training_df[AWAY_FEATURES])

    home_model = sm.GLM(training_df['actual_home'], X_home, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(training_df['actual_away'], X_away, family=sm.families.Poisson()).fit()

    return home_model, away_model


# %% [markdown]
# ### Extracting features for a match to be predicted
#
# Same existence/data-sufficiency checks and league-average fallbacks as
# `expected_lambdas()` in 02_match_level_model.py, extended to also fall
# back to a league-average referee rate when the referee is new or has
# officiated too few matches.

# %%
def get_match_features(
    home_id, away_id, ref_id, teams, refs,
    league_home_avg_cards, league_away_avg_cards, league_ref_avg_cards,
    min_team_matches=3, min_ref_matches=3,
):
    # NaN comparisons are always False, so a team that has played on the
    # other side but never this one (e.g. promoted, away matches only so
    # far) needs an explicit isna() check or it slips past the fallback.
    if (
        home_id not in teams.index
        or pd.isna(teams.loc[home_id, 'home_matches_count'])
        or teams.loc[home_id, 'home_matches_count'] < min_team_matches
    ):
        home_for = league_home_avg_cards
        home_against = league_away_avg_cards
    else:
        home_for = teams.loc[home_id, 'home_cards_for']
        home_against = teams.loc[home_id, 'home_cards_against']

    if (
        away_id not in teams.index
        or pd.isna(teams.loc[away_id, 'away_matches_count'])
        or teams.loc[away_id, 'away_matches_count'] < min_team_matches
    ):
        away_for = league_away_avg_cards
        away_against = league_home_avg_cards
    else:
        away_for = teams.loc[away_id, 'away_cards_for']
        away_against = teams.loc[away_id, 'away_cards_against']

    if ref_id not in refs.index or refs.loc[ref_id, 'ref_match_count'] < min_ref_matches:
        ref_cards_per_game = league_ref_avg_cards
    else:
        ref_cards_per_game = refs.loc[ref_id, 'ref_cards_per_game']

    return {
        'home_for': home_for, 'home_against': home_against,
        'away_for': away_for, 'away_against': away_against,
        'ref_cards_per_game': ref_cards_per_game,
    }


# %% [markdown]
# ### Poisson log-loss (identical to 02_match_level_model.py)

# %%
def poisson_log_loss(results_df, predicted_col='predicted_cards', actual_col='actual_cards'):
    probabilities = poisson.pmf(results_df[actual_col], results_df[predicted_col])
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    return -np.log(probabilities).mean()


def calculate_overall_metrics(results_df, predicted_col='predicted_cards', actual_col='actual_cards'):
    error = results_df[predicted_col] - results_df[actual_col]
    return {
        'matches': len(results_df),
        'mae': error.abs().mean(),
        'rmse': np.sqrt((error ** 2).mean()),
        'bias': error.mean(),
        'log_loss': poisson_log_loss(results_df, predicted_col, actual_col),
        'within_2': (error.abs() <= 2).mean() * 100,
    }


def print_metrics(metrics, title):
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    print(f"Matches:        {metrics['matches']:,}")
    print(f"MAE:            {metrics['mae']:.4f}")
    print(f"RMSE:           {metrics['rmse']:.4f}")
    print(f"Bias:           {metrics['bias']:.4f}")
    print(f"Log-Loss:       {metrics['log_loss']:.4f}")
    print(f"Within 2 cards: {metrics['within_2']:.1f}%")
    print("=" * 60)


# %% [markdown]
# ## Walk-Forward Loop
#
# Same discipline as the production model: train on the earliest season,
# then step forward week by week, refitting the Poisson GLM on everything
# known up to that point (never on the week being predicted).

# %%
def run_poisson_walk_forward(match_data, min_training_rows=100):
    first_season = sorted(match_data['season'].unique())[0]
    history = match_data[match_data['season'] == first_season].copy()
    future = match_data[match_data['season'] != first_season].copy()
    test_weeks = sorted(future['week'].unique())

    print("Initial training season:", first_season)
    print("Number of test weeks:", len(test_weeks))

    team_history = []
    ref_history = []
    results = []

    for week in test_weeks:
        batch = future[future['week'] == week].copy()
        asof_date = batch['Date'].min()

        history_with_weights = add_time_weights(history, asof_date)

        league_home_avg_cards = wmean(history_with_weights['home_cards'], history_with_weights['weight_team'])
        league_away_avg_cards = wmean(history_with_weights['away_cards'], history_with_weights['weight_team'])
        league_ref_avg_cards = wmean(history_with_weights['total_cards'], history_with_weights['weight_ref'])

        teams = build_team_profiles(history_with_weights)
        refs = build_ref_profiles(history_with_weights)

        teams_snapshot = teams.copy()
        teams_snapshot['team_id'] = teams_snapshot.index
        teams_snapshot['week'] = week
        team_history.append(teams_snapshot)

        refs_snapshot = refs.copy()
        refs_snapshot['referee_id'] = refs_snapshot.index
        refs_snapshot['week'] = week
        ref_history.append(refs_snapshot)

        teams_over_time = pd.concat(team_history, ignore_index=True)
        refs_over_time = pd.concat(ref_history, ignore_index=True)

        training_df = build_glm_training_table(history, teams_over_time, refs_over_time, min_rows=min_training_rows)

        if len(training_df) < min_training_rows:
            print(f"Week {week}: only {len(training_df)} training rows, skipping (need {min_training_rows}).")
            history = pd.concat([history, batch], ignore_index=True)
            continue

        home_model, away_model = fit_poisson_models(training_df)

        for _, row in batch.iterrows():
            features = get_match_features(
                home_id=row['home_team_id'], away_id=row['away_team_id'], ref_id=row['referee_id'],
                teams=teams, refs=refs,
                league_home_avg_cards=league_home_avg_cards,
                league_away_avg_cards=league_away_avg_cards,
                league_ref_avg_cards=league_ref_avg_cards,
            )

            X_home = pd.DataFrame([{'const': 1.0, **{f: features[f] for f in HOME_FEATURES}}])[['const'] + HOME_FEATURES]
            X_away = pd.DataFrame([{'const': 1.0, **{f: features[f] for f in AWAY_FEATURES}}])[['const'] + AWAY_FEATURES]

            predicted_home = home_model.predict(X_home).iloc[0]
            predicted_away = away_model.predict(X_away).iloc[0]

            results.append({
                'Date': row['Date'],
                'week': week,
                'match_id': row['match_id'],
                'actual_cards': row['total_cards'],
                'actual_home': row['home_cards'],
                'actual_away': row['away_cards'],
                'predicted_home': predicted_home,
                'predicted_away': predicted_away,
                'predicted_cards': predicted_home + predicted_away,
            })

        history = pd.concat([history, batch], ignore_index=True)

    return pd.DataFrame(results)


# %% [markdown]
# ## Run the comparison

# %%
if __name__ == '__main__':
    match_data = load_match_data()
    poisson_results = run_poisson_walk_forward(match_data)
    poisson_results = poisson_results.sort_values('Date').reset_index(drop=True)

    poisson_metrics = calculate_overall_metrics(poisson_results)
    print_metrics(poisson_metrics, "POISSON REGRESSION (GLM) PERFORMANCE")

    # Load the production model's already-saved predictions for comparison
    existing_results = pd.read_csv('outputs/results/match_predictions.csv')

    # Compare on the matches both models actually predicted
    common_ids = set(poisson_results['match_id']) & set(existing_results['match_id'])
    print(f"\nMatches predicted by both models: {len(common_ids)}")

    existing_common = existing_results[existing_results['match_id'].isin(common_ids)]
    poisson_common = poisson_results[poisson_results['match_id'].isin(common_ids)]

    existing_metrics = calculate_overall_metrics(existing_common)
    poisson_metrics_common = calculate_overall_metrics(poisson_common)

    print_metrics(existing_metrics, "EXISTING MODEL (common matches)")
    print_metrics(poisson_metrics_common, "POISSON GLM (common matches)")

    comparison = pd.DataFrame([
        {'model': 'Existing (correlation-weighted)', **existing_metrics},
        {'model': 'Poisson regression (GLM)', **poisson_metrics_common},
    ])
    print("\n" + comparison.to_string(index=False))

    os.makedirs('outputs/results', exist_ok=True)
    poisson_results.to_csv('outputs/results/poisson_match_predictions.csv', index=False)
    comparison.to_csv('outputs/results/poisson_vs_existing_comparison.csv', index=False)
    print("\nSaved outputs/results/poisson_match_predictions.csv")
    print("Saved outputs/results/poisson_vs_existing_comparison.csv")
