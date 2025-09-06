# Fantasy Football Data Analysis Project 🏈

The beginning of the NFL season also marks the start of the fantasy football season. Like clockwork, the experts are out in full force to tell you which players to draft and which round to take them in. After a few seasons of playing fantasy football, I decided to dive head first into the actual data to draw conclusions for myself. In this python project I use real NFL data from 2017 - 2024 to investigate patterns, find common trends, and create a model using machine learning to forecast the 2025 season.

## Data 📊

1. Pro Football Reference (2017-2024), NFL Fantasy Rankings, <https://www.pro-football-reference.com/years/2024/fantasy.htm>
2. fantasydata (2017-2024), Snap Counts, <https://fantasydata.com/nfl/nfl-snap-counts>

Several data sets are used in this project from both Pro Football Reference and fantasydata.com. The former provides general statistics for players such as rushing yards, completions, touchdowns, etc… while the latter provides snap count data. When first beginning this project I hypothesized that overall player usage such as targets, rush attempts, and time on the field would be the statistics that are most predictive of future fantasy football success. To further explore this, I include snap count data which is provided by fantasydata.com.

## Cleaning 🧹

The goal of this step is to merge all of the data frames with as few issues as possible. To do this, several steps need to be taken such as removing unnecessary columns, filling null values where they can be filled, and removing those rows with null values where applicable. Without diving too deep into this process, here are a few of the measures taken.

- Creating functions to standardize each data set:
Below is an example of a function used to clean the pro football reference data

```python
def clean_FFBD(df):
    df = df[~(df[['Cmp', 'Att (Pass)', 'Tgt']].isna().all(axis=1))]
    df = df.drop_duplicates()
    df[['2PM', '2PP']] = df[['2PM', '2PP']].fillna(0.0)
    df.index = df.index.str.rstrip('*+')
    df.index = df.index.str.replace(r'\sJr\.$', '', regex = True)
    df.index = df.index.str.replace(r'\sSr\.$', '', regex = True) 
    df.index = df.index.str.replace(r'\.', '', regex = True)
    df.index = df.index.str.replace(r'\sIII$', '', regex = True)
    df.index = df.index.str.replace(r'\sII$', '', regex = True)
    df.index = df.index.str.replace(r'\sIV$', '', regex = True)
    df.index = df.index.str.replace(r'\sV$', '', regex = True)
    df.index = df.index.str.upper().str.strip()
    df = df[df['FantPos'] != 'FB']
    return df
```

- Merging each individual year of data, then merging the general player data with snap count data

```python
FFBD_Merged = pd.concat([FFBD2024, FFBD2023, FFBD2022, FFBD2021, FFBD2020, FFBD2019, FFBD2018, FFBD2017])
FFBD_tot = pd.merge(FFBD_Merged, FFBDsnp_Merged, left_index=True, right_index=True, how = 'inner')
```

- Creating custom insights not provided in the original data:
Below is an example of how we can create the passer rating statistic for QBs

```python
mask = (FFBD_tot['Att (Pass)'] !=0) & (pd.notna(FFBD_tot['Att (Pass)']))

FFBD_tot.loc[mask, 'Y/A_rate'] = ((FFBD_tot.loc[mask, 'Yds (Pass)'] / FFBD_tot.loc[mask, 'Att (Pass)']) - 3) * 5
FFBD_tot.loc[mask, 'Cmp_rate'] = ((FFBD_tot.loc[mask, 'Cmp'] / FFBD_tot.loc[mask, 'Att (Pass)']) - 0.3) * 0.25
FFBD_tot.loc[mask, 'TD_rate'] = (FFBD_tot.loc[mask, 'TD (Pass)'] / FFBD_tot.loc[mask, 'Att (Pass)']) * 20
FFBD_tot.loc[mask, 'Int_rate'] = 2.375 - (FFBD_tot.loc[mask, 'Int'] / FFBD_tot.loc[mask, 'Att (Pass)']) * 25

Rates = ['Y/A_rate', 'Cmp_rate', 'TD_rate', 'Int_rate']
FFBD_tot[Rates] = FFBD_tot[Rates].clip(lower=0, upper=2.375)
    
FFBD_tot['Passer_Rating'] = (FFBD_tot['Y/A_rate'] + FFBD_tot['Cmp_rate'] + FFBD_tot['TD_rate'] + FFBD_tot['Int_rate'])/6 * 100

FFBD_tot.drop(columns=Rates, inplace=True)
```

## Exploratory Data Analysis 🔍

To start the EDA process, I made some general observations about the value of each position. Additionally, I observed which NFL teams scored the most fantasy points in 2024 trhough grouping the data by team and the sum of PPR points scored.
When sorting the data by the top fantasy PPR finishes since 2017, we see that quarterbacks account for the majority of the top spots. It is important to note that despite this majority, the number one overall fantasy season came from Christian McCaffrey – a running back – in 2023.
Playoff teams and high powered offenses score the most fantasy points as a unit e.g. Tampa Bay, Detroit, Baltimore. A team that finished on the outside looking in last season was the Cincinnati Bengals, but according to the bar plot they actually scored the third most overall fantasy points. Since they didn't make many additions to their defense this season, look for them to have to put up a lot of points again in 2025.
The findings are illustrated in the figures below:

![Top Finishes Pie Graph](/Images/Top_50_finishes_split.png)

![NFL Teams PPR](/Images/PPR_Teams_2024.png)

Investigating the data on a per position basis, we  can learn about what stats are most indicative of a star fantasy player. The following figures visualize the top NFL players for fantasy in 2024...

Begining with QBs, we can take a look at some of the efficiency metrics created post data cleaning. Taking the top ten quarterbacks from the 2024 season; passing efficiency, rushing efficiency, and touchdown percentage (touchdowns/snaps) can be plotted over the past seven seasons

![QB Efficiency Last 7 seasons](/Images/Top_QBs_line.png)

Arguably known more for his rushing prowess, Lamar Jackson leads this group in PPR points scored per throw in recent years. Jared Goff – a pure pocket passer – lands in second through the same two seasons.
In terms of rushing efficiency, Josh Allen finishes with the majority of first place years with three, including 2023 and 2024. This indicates that on a per rush basis, he tends to rush for the most yards while also scoring touchdowns at a relatively high rate.
Speaking of touchdown percentage, Jalen Hurts leads the pack in this category with 1.5% of his snaps resulting in touchdowns. This makes a lot of sense considering the Eagles have a play that is nearly always run from the one yardline which results in QB touchdowns (the brotherly shove).

Taking a look at the RB position, we can visualize the best fantasy backs from a year ago along with their usage.
The first plot in the following figure visualizes the RBs with the most PPR points scored solely via running the football. In that category Derrick Henry finished as the RB #1, but lacked meaningful usage through the air. Inversely when looking at the top producer among RB fantasy points through the air, we see that De'Von Achane had a relatively low PPR finish running the football.
My personal takeaway from this is that the most valuable backs this upcoming season could be the players appearing on both plots; Jahmyr Gibbs and Bijan Robinson are the most balanced players in this upper echelon of backs.

![Top RBs 2024](/Images/Top_RBs_2024.png)

One of the crucial differences between non-PPR and full PPR formats is that the latter awards one point for every catch. It may not seem like much, but it can skew positional value to players who catch a lot of passes; such is the case with WRs. Since both one catch and ten yards each translate to one PPR fatasy point, I hypothesized that there would be enough WRs catching passes for short gains (less than ten yards) that the majority of points for receivers on the whole would come from catches. Looking at the WR data from a birds eye view, this theory proved to be incorrect, with 44% of non-TD points coming from receptions and 56% from yards gained.

![WR Point Pie](/Images/PPR_split_WRs.png)

As we'll soon see from the models created for predicting fantasy points, the stickiest stats for WRs are yards, receptions, and targets. To investigate which players have been proficient in both volume as well as efficiency, I plotted the top ten WRs by fantasy finish for 2024. The following bar plot displays the receiver's PPR points on the y-axis as well as their efficiency (PPR/target) and volume (Number of targets) depicted through a color gradient:

![WR PPR Top Ten](/Images/Top_WRs_2024.png)

One receiver that stands out to me in the above bar plot is Terry McLaurin. He had similar efficiency to Ja'marr Chase (the #1 WR in fantasy a year ago). If you have reason to believe McLaurin will get more targets next season, it's possible that he could finish closer toward the top of this list. In contrast, Malik Nabers' season saw him get peppered with targets while recording lower relative efficiency. Considering the volatility at the QB position for the Giants in 2024, I wouldn't be too concerned with Nabers' 2025 outlook. In fact, my prediciton models have Nabers finishing first among WRs this upcoming season.

Since TEs record fantasy points in much the same manner as WRs, I produced a similar bar plot with the top ten TEs from the 2024 season:

![Top TEs PPR](/Images/Top_TEs_2024.png)

The one player that stands out to me from this visualization is Mark Andrews. It was just four seasons ago that Andrews was the best TE from a fantasy perspective, but with backup TE Isiah Likely getting targets in 2024, Andrews' overall production decreased. If the Ravens go back to featuring Andrews in a larger role in 2025, look for him to return to the top three fantasy TE conversation.

## Creating XGBoost Models to Predict Next Season's Fantasy Point Finishes

Methodology:
The XGBoost algorithm allows us to train and test predictive models using historical player data. Please note that these models are not perfect and should be taken with a grain of salt. With all of the noise which affects football stats every year (injuries, weather conditions, coaching, etc...) there is only so much variance that can be explained by the models. For full training/testing root mean squared errors and r squared values, please see the jupyter notebook.

Features:
For each model we can create a few extra features in addition to the data provided from Pro Football Reference and the features created for the exploratory analysis. These include career averages, lagged stats for the previous three seasons, and the 'PPR_next_year' column, which will be the figure that the models are predicting.

QB Features Excerpt:

```python
QBs['Career_Pass_Att'] = QBs.groupby('Player')['Att_Pass'].cumsum().shift(1)
QBs['Career_Pass_Att'] = QBs['Career_Pass_Att'].fillna(0)
```

Training/Testing Data:
By separating the historical data into three groups, we can train and test the models as well as make predictions. For each model, 2017-2022 data was used to train, 2023 data was used to test, and 2024 allowed for the model to make 2025 predictions.

QBs Train/Test Excerpt:

```python
QBs_train = QBs[QBs['Year'] < 2023].dropna(subset=['PPR_next_year'])
QBs_val = QBs[QBs['Year'] == 2023].dropna(subset=['PPR_next_year'])
QBs_test = QBs[QBs['Year'] == 2024]
```

Reviewing feature importances:
In order to create the finalized models, feature importances were examined to ensure certain features were not improperly fitting the model. For example, lagged touchdowns were removed in the TE model for being too predictive given their high volatility from season to season.

The top three feature importances in each model are as follows...

QBs:

- Passing Yards (1-year lagged) : 0.273659
- Completions (1-year lagged) : 0.107511
- Pass Attempts (1-year lagged) : 0.063374

RBs:

- Rushing Yards (1-year lagged) : 0.204586
- Touchdowns (1-year lagged) : 0.095579
- Yards Receiving (1-year lagged) : 0.081882

WRs:

- Yards Receiving (1-year lagged) : 0.322328
- Receptions (1-year lagged) : 0.204282
- Targets (1-year lagged) : 0.064668

TEs:

- Yards Receiving (1-year lagged) : 0.219625
- Targets (1-year lagged) : 0.201992
- Receptions (1-year lagged) : 0.095219

Unsurprisingly the features with the highest importance to the models are consistently the previous season's statistics. In the case of WRs and TEs, the models have the same top three feature importances, with the order of targets and receptions being flipped. It's likely that targets were more predictive for TEs than receptions since it also captures volume in the passing game. Often TEs will be deployed as blockers rather than receivers, generally leading to a lower volume of work in the passing game.

Results:
The following figures represent the top fifteen PPR finishes at every position for the 2025 season as predicted by the XGBoost models.

![QB Predictions](/Images/QB_Predict_2025.png)
![RB Predictions](/Images/RB_Predict_2025.png)
![WR Predictions](/Images/WR_Predict_2025.png)
![TE Predictions](/Images/TE_Predict_2025.png)

Results Summary:
There are a few players that stood out as being unlikely to finish in the spot that the model predicted them to finish in.
First, Jared Goff is unlikely to finish second in fantasy points. It's not impossible, but he does not have the same rushing value as the other names around him. It would take an MVP type season through the air for him to come close to Daniels, Jackson, and Allen.
Secondly, Rachaad White is unlikely to finish in the top five for running backs. I imagine the model is taking his volume from the past few seasons into account, but in 2025 he is projected to be the backup to Bucky Irving (the player just behind White in the predictions).
The Wide Receiver and Tight End predictions look mostly accurate to my eye, though Jordan Addison's position is a tad bit rich (he is suspended three games and is behind another star receiver on the depth chart).

A common thread across each model's predictions is that second year players have dramatic increases in production compared to their rookie seasons. It's certainly common for NFL players to take a leap forward in their second year, and in the case of players with already stellar rookie seasons such as Brian Thomas and Malik Nabers, it's feasible that they could indeed finish in the top three in fantasy points. Jayden Daniels is also predicted to finish at the top spot for QBs in his second outing as well as Brock Bowers for TEs.

Overall I am pleased with the predictions of each of the models given their consistency with NFL trends (i.e. second year players breaking out) and with recently recorded player statistics. Upon the conclusion of the 2025 fantasy football season, I look forward to reviewing the predictions once again and comparing the errors with those from the 2023 validation data.

Thank you for reaching the end of this readme! Hopefully you've gained some valuable information for the upcoming season. If you choose to share any of my work with your fantasy football colleagues on the web, I only ask that you please give me a citation. Good luck fantasy footballers!
