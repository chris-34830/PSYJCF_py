import logging
import QuantLib as ql
from ..utils.logging_utils import log_function_call

def calibrate_heston_model_from_iv(
    eval_date,
    spot_price,
    rf_rate,
    dividend_rate,
    df_options,
    iv_col="implied_vol_BS",
    init_params=None  
):
    """
    Calibrate the Heston model using market implied volatilities.
    
    Parameters:
    -----------
    eval_date : datetime.date
        Evaluation date
    spot_price : float
        Current stock price
    rf_rate : float
        Risk-free interest rate
    dividend_rate : float
        Continuous dividend yield
    df_options : pandas.DataFrame
        DataFrame containing option data
    iv_col : str
        Column name containing implied volatilities to calibrate against
    init_params : QuantLib.Array, optional
        Initial parameters for the optimization
    """
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql.Date(
        eval_date.day, 
        eval_date.month, 
        eval_date.year
    )

    # Setup yield curves
    risk_free_curve = ql.FlatForward(
        ql.Settings.instance().evaluationDate, 
        rf_rate,
        day_count,
        ql.Compounded,
        ql.Continuous
    )
    risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)

    dividend_curve = ql.FlatForward(
        ql.Settings.instance().evaluationDate, 
        dividend_rate,
        day_count,
        ql.Compounded,
        ql.Continuous
    )
    dividend_handle = ql.YieldTermStructureHandle(dividend_curve)

    # Initial Heston parameters if not provided
    if init_params is None:
        v0 = 0.01      # Initial variance
        kappa = 1.0    # Mean reversion speed
        theta = 0.01   # Long-term variance
        sigma = 0.2    # Volatility of variance
        rho = -0.5     # Correlation
    else:
        kappa = float(init_params[0])
        theta = float(init_params[1])
        sigma = float(init_params[2])
        rho = float(init_params[3])
        v0 = float(init_params[4])

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    
    # Create Heston process and model
    heston_process = ql.HestonProcess(
        risk_free_handle,
        dividend_handle,
        spot_handle,
        v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)

    # Create calibration helpers
    helpers = []
    for idx, row in df_options.iterrows():
        if not row.get("included", False):
            continue
            
        implied_vol = row.get(iv_col, None)
        if implied_vol is None or pd.isna(implied_vol) or implied_vol <= 0:
            continue

        T = row.get("T", 0)
        if T <= 0:
            continue

        K = row.get("strike", 0)
        if K <= 0:
            continue

        vol_handle = ql.QuoteHandle(ql.SimpleQuote(implied_vol))
        maturity_days = int(365 * T)
        if maturity_days < 1:
            continue

        helper = ql.HestonModelHelper(
            ql.Period(maturity_days, ql.Days),
            calendar,
            spot_price,
            K,
            vol_handle,
            risk_free_handle,
            dividend_handle,
            ql.BlackCalibrationHelper.ImpliedVolError
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    if len(helpers) == 0:
        logging.warning(f"Aucun helper créé pour la calibration (iv_col={iv_col}).")
        return None, None

    # Calibration using Levenberg-Marquardt algorithm
    optimization_method = ql.LevenbergMarquardt()
    end_criteria = ql.EndCriteria(1000, 100, 1.0e-8, 1.0e-8, 1.0e-8)
    
    try:
        heston_model.calibrate(
            helpers,
            optimization_method,
            end_criteria
        )

        # Log calibrated parameters
        logging.info(
            f"[HESTON-{iv_col}] Paramètres calibrés:\n"
            f"kappa={heston_model.kappa():.4f}, "
            f"theta={heston_model.theta():.4f}, "
            f"sigma={heston_model.sigma():.4f}, "
            f"rho={heston_model.rho():.4f}, "
            f"v0={heston_model.v0():.4f}"
        )

        return heston_model, heston_process

    except Exception as e:
        logging.error(f"Erreur lors de la calibration Heston: {str(e)}")
        return None, None

@log_function_call
def price_european_option_heston(
    heston_model,
    strike,
    maturity_in_years,
    option_type="C"
):
    """
    Price a European option using the calibrated Heston model.
    
    Parameters:
    -----------
    heston_model : ql.HestonModel
        Calibrated Heston model
    strike : float
        Strike price
    maturity_in_years : float
        Time to maturity in years
    option_type : str
        Option type ('C' for call, 'P' for put)
    
    Returns:
    --------
    float : Option price
    """
    try:
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type.upper() == "C" else ql.Option.Put,
            strike
        )
        
        eval_date = ql.Settings.instance().evaluationDate
        maturity_date = eval_date + int(365 * maturity_in_years)
        exercise = ql.EuropeanExercise(maturity_date)
        
        european_option = ql.VanillaOption(payoff, exercise)
        engine = ql.AnalyticHestonEngine(heston_model)
        european_option.setPricingEngine(engine)
        
        return european_option.NPV()
        
    except Exception as e:
        logging.error(f"Erreur lors du pricing Heston: {str(e)}")
        return None
    