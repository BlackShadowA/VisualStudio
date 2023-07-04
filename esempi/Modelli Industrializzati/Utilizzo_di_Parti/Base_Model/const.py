FEATURES_WEIGHT = {
    'transaction__': 80,  # previously, all 5
    'tfa__': 25,
    'bancass__': 30,
    'cntp__': 25,
    'advice__': 20,
    'loan__': 30,
    'margin__': 25,
    'account__': 25,
    'deal__': 25,
    'rel__': 25,
    'securities__': 20,
    'claim__': 15,
}

# features that will be add on top at the feature selection phase
HARDCODED_FEATURES = [
    "subscription__insurance_monthly_count_keyword",
    "subscription__insurance_monthly_sum_amount_keyword",
    "subscription__insurance_fl_match",
    "subscription__insurance_month_number",
    "subscription__insurance_count_monthly_keyword_l12m",
    "subscription__insurance_mean_stddev_monthly_amount_keyword_l12m",
    "subscription__insurance_mean_avg_monthly_amount_keyword_l12m",
    "subscription__insurance_sum_amount_keyword_l12m",
    "subscription__insurance_count_monthly_keyword_l3m",
    "subscription__insurance_mean_stddev_monthly_amount_keyword_l3m",
    "subscription__insurance_mean_avg_monthly_amount_keyword_l3m",
    "subscription__insurance_sum_amount_keyword_l3m",
    "subscription__insurance_mean_cv_monthly_amount_keyword_l12m",
    "subscription__insurance_mean_cv_monthly_amount_keyword_l3m",
    "subscription__insurance_subscription",
    "subscription__insurance_yearly_subscription"
]