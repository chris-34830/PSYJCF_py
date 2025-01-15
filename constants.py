# Option types
OPTION_TYPES = {
    'CALL': 'C',
    'PUT': 'P'
}

# Column names for different data categories
COLUMN_NAMES = {
    'price': {
        OPTION_TYPES['CALL']: 'call_price',
        OPTION_TYPES['PUT']: 'put_price'
    },
    'vol_models': [
        'implied_vol_CRR',
        'implied_vol_BS',
        'implied_vol_BAW'
    ],
    'market_data': [
        'bid',
        'ask',
        'lastPrice',
        'volume',
        'openInterest'
    ],
    'computed': [
        'market_price',
        'moneyness',
        'included',
        'exclusion_reason'
    ]
}