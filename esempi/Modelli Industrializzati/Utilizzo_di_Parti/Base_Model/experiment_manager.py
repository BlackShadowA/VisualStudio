from cards_propensity_experiments.utils.benchmarks import BaselineModels
from typing import Dict, Any

EXPERIMENT_NAME = 'baseline'
BASE_PATH = "/uci/dp_marketing_models/technical/apps/Cards/Individuals/data/train"
TARGET_VARIABLE = 'target'
N_Q = 100


BASELINE_MODELS = {
    "lightgbm": BaselineModels.get_lgb,
    "balanced_lgb": BaselineModels.get_lgb_balanced,
    "goss_lgb": BaselineModels.get_lgb_goss,
    "goss_balanced_lgb": BaselineModels.get_lgb_goss_balanced,
    "gb_rf_balanced_xgb": BaselineModels.get_xgb_gb_rf_balanced,
    "goss_xgb": BaselineModels.get_xgb_gb,
    "svm": BaselineModels.get_svc,
    "catboost": BaselineModels.get_cb,
    "catboost_mvs": BaselineModels.get_cb_mvs_balanced,
    "random_forest": BaselineModels.get_random_forest
}

FEATURE_SELECTION_BASELINE = [
    # CARDS
    'F070026_card__credit_opened_l12m_num',
    'F070020_card__credit_months_from_last_activation',
    'F070091_card__debit_opened_l12m_num',
    'F070115_card__prepaid_opened_l12m_num',
    'F070134_card__months_from_last_activation',
    # CATEGORIZER 
    'F037097_transaction__uci_categorizer_cat_transports_and_travels_amount_deb_l12m_num',
    'F037127_transaction__uci_categorizer_tp_atm_and_withdrawals_amount_deb_l12m_num',
    'F037115_transaction__uci_categorizer_cat_vehicles_amount_deb_l12m_num',
    'F037225_transaction__uci_categorizer_tp_subscriptions_amount_deb_l12m_num',
    'F037079_transaction__uci_categorizer_cat_sport_and_free_time_amount_deb_l12m_num',
    'F037234_transaction__uci_categorizer_tp_virtual_wallets_amount_deb_l12m_num',
    # TFA
    'F050023_tfa__direct_deposits_balance_monthlyagg_l12m_mean',
    'F050041_tfa__asset_under_mgmt_balance_monthlyagg_l12m_mean',
    'F050032_tfa__asset_under_custody_balance_monthlyagg_l12m_mean',
    # DEALS
    'F007541_deal__active_l12m_num',
    'F007552_deal__genius_cards_prepaid_active_num',
    # IBLO 
    'F125107_iblo__total_ops_l6m_num',
    'F125032_iblo__mobile_ops_l6m_num', #F125108_iblo__mobile_ops_l6m_over_total_ops_l6m_percentage
    'F125034_iblo__web_ops_l6m_num', #F125109_iblo__web_ops_l6m_over_total_ops_l6m_percentage
    'F125036_iblo__assisted_ops_l6m_num', #F125110_iblo__assisted_ops_l6m_over_total_ops_l6m_percentage
    'F125111_iblo__total_informative_ops_l6m_num', #F125112_iblo__informative_ops_l6m_over_total_ops_l6m_percentage
    'F125038_iblo__mobile_provisioning_ops_l6m_num', #F125114_iblo__provisioning_ops_l6m_over_total_ops_l6m_percentage
    'F125040_iblo__web_provisioning_ops_l6m_num',
    # CURRENT ACCOUNT
    'F020002_account__accounting_balance_sum',
    'F020015_account__accounting_balance_monthlyagg_l12m_mean',
    'F020020_account__accounting_balance_monthlyagg_l12m_linear_trend_attribute_slope',
    # TRANSACTION
    'F035201_transaction__uci_vol_l12m_num',
    'F035200_transaction__uci_cred_l12m_num',
    'F035199_transaction__uci_deb_l12m_num',
    'F035163_transaction__uci_sdd_amount_deb_monthlyagg_l12m_mean',
]


################################################################## EXPERIMENT DICIONARY ##################################################################

config_experiment: Dict[str, Any] = {
        'standard': {
                "model": BASELINE_MODELS,
                "train": f"{BASE_PATH}/train_test",
                "output_location": f"{BASE_PATH}/model"
        }
    }