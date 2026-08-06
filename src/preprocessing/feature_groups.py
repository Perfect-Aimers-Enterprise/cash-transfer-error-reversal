"""
Feature Groups for Ablation Study

Each group can be removed independently during
training to evaluate its contribution to the model.

Example

BASE_FEATURES

TEMPORAL_FEATURES

CHANNEL_FEATURES

Remove one group

features = (
    BASE_FEATURES
    + BEHAVIOURAL_FEATURES
)

"""

# ==========================================================
# Base Transaction Features
# ==========================================================

BASE_FEATURES = [

    "amount",

    "amount_log",

]

# ==========================================================
# Temporal Features
# ==========================================================

TEMPORAL_FEATURES = [

    "timestamp",

    "hour",

    "is_night",

    "is_weekend",

]

# ==========================================================
# Sender Behaviour
# ==========================================================

SENDER_FEATURES = [

    "sender_txn_count",

    "sender_avg_amount_so_far",

    "sender_std_amount_so_far",

    "sender_seconds_since_last",

    "sender_error_rate_so_far",

]

# ==========================================================
# Beneficiary Behaviour
# ==========================================================

BENEFICIARY_FEATURES = [

    "beneficiary_txn_count_so_far",

    "beneficiary_distinct_senders_so_far",

    "is_new_beneficiary_for_sender",

]

# ==========================================================
# Device / Location Behaviour
# ==========================================================

DEVICE_LOCATION_FEATURES = [

    "is_new_device_for_sender",

    "is_new_location_for_sender",

]

# ==========================================================
# Duplicate Detection
# ==========================================================

DUPLICATE_FEATURES = [

    "is_duplicate_like",

]

# ==========================================================
# Channel Features
# ==========================================================

CHANNEL_FEATURES = [

    "channel_mobile",

    "channel_web",

    "channel_ussd",

]

# ==========================================================
# Other Features
# ==========================================================

OTHER_FEATURES = [

    "location",

    "reversal_reason",

    "reversal_executed",

]

# ==========================================================
# Complete Feature Set
# ==========================================================

ALL_FEATURES = (

    BASE_FEATURES

    + TEMPORAL_FEATURES

    + SENDER_FEATURES

    + BENEFICIARY_FEATURES

    + DEVICE_LOCATION_FEATURES

    + DUPLICATE_FEATURES

    + CHANNEL_FEATURES

    + OTHER_FEATURES

)

TARGET = "error_flag"

# ==========================================================
# Ablation Configurations
# ==========================================================

ABLATION_GROUPS = {

    "all_features": ALL_FEATURES,

    "without_temporal": [
        f for f in ALL_FEATURES
        if f not in TEMPORAL_FEATURES
    ],

    "without_sender": [
        f for f in ALL_FEATURES
        if f not in SENDER_FEATURES
    ],

    "without_beneficiary": [
        f for f in ALL_FEATURES
        if f not in BENEFICIARY_FEATURES
    ],

    "without_device_location": [
        f for f in ALL_FEATURES
        if f not in DEVICE_LOCATION_FEATURES
    ],

    "without_duplicate": [
        f for f in ALL_FEATURES
        if f not in DUPLICATE_FEATURES
    ],

    "without_channel": [
        f for f in ALL_FEATURES
        if f not in CHANNEL_FEATURES
    ],

    "without_base": [
        f for f in ALL_FEATURES
        if f not in BASE_FEATURES
    ],

}