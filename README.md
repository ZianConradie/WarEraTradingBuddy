# WarEraTradingBuddy
A Python app that predicts future prices for the warera.io stock market, and automatically notifies you through "ntfy" if a trading oppurtunity appears.

# How to get started
This really is the bare minimum to get started, can some collaborators pls make this stuff better, thanks

# MOBILE Setup:
1. Download the "ntfy" app.
2. Press the plus button (if applicable)
3. By Topic Name, enter anything you want. This will kind of be the "ID" of your project.
   I suggest calling it something reasonable, and adding 10 random digits at the end.
   While not strictly neccecary, this helps so that other people won't get your notifications.
5. Press "Subscribe"

# PC Setup:

## 1. Run "download_history.py"
This file will download all price history data from present day to as far back as he can and save it in /price_history/

## 2. Edit "train.py"
At line 48-50, you will see GRU settings:

```
# GRU settings.
GRU_HIDDEN_SIZE = 128
GRU_LAYERS = 2
```

These are the most basic settings of your model that will be predicting the prices.
The better your computer, the higher numbers you can use here, but I'm running a 13th gen i3 CPU, and default settings work.

## 3. Run "train.py"
Depending on your settings and computer, this might take a while (or not).
All this will do is train a model for each item in warera and save them in /price_models/

## 4. Edit "main.py"
This is what will allow you to recieve the notifications on your phone.
There are 4 main things you'll edit on line 47-63.

```
# ============================================================
# CONFIGURATION
# ============================================================

TOPIC = "WarEraTrading-********"

NOTIFICATION_PRIORITY = 3 # popup, no vibration, no sound

# Number of future days to predict.
PREDICTION_DAYS = 2+1
MIN_HISTORY_DAYS = 30

# Number of real historical days used to determine trend.
TREND_DAYS = 2

# Only send notifications at or above this confidence.
MIN_CONFIDENCE = 0.95
```
"TOPIC" is whatever name you chose earlier. It MUST be exactly the same.

"NOTIFICATION_PRIORITY" handles how the notification gets sent to your phone.
1-3: No Sound, No Vibration, No Popup
4: No Sound, Short Vibration, Popup
5: Sound, long Vibration, popup.

3 is default.

"PREDICTION_DAYS" is how far out the script will look for price reversals.
Default is 2+1, meaning it will look 2 days into the future to look for price reversals.

"MIN_CONFIDENCE" is the decimal of how confident the model is about the price reversal.
Default is 0.95, meaning the model must be at least 95% sure the reversal is going to happen before sending a notification.

## 5. Run "main.py"
The script should immidietely search for trading oppurtunities, and send a notification if any oppurtunities appear.
This will happen once every hour, on the hour.


# KNOWN ISSUES
- main.py automatically fetches the latest price, if no prices has been saved for that day, meaning at midnight every day, the script will fetch those items at midnight, and doesnt update the prices for the rest of the day.
- The dataset that we download the prices from doesnt have red cases for some reason.

# TODO List
- Maybe related items  such as iron and steel should have eachover's prices as context / input. Maybe that will help?
