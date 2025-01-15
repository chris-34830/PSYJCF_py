import datetime
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import config
from utils.logging_utils import setup_logging
from data.fetcher import fetch_option_chain
from data.processor import OptionDataProcessor
from viz.plotter import plot_volatility_surface
from models.heston import (
    calibrate_heston_model_from_iv,
    price_european_option_heston
)
from constants import OPTION_TYPES, COLUMN_NAMES

class OptionAnalyzer:
    """Classe principale pour l'analyse des options financières."""
    
    def __init__(self):
        """Initialisation des paramètres et configuration du logging."""
        # Configuration du système de logging
        setup_logging(config)
        self.logger = logging.getLogger(__name__)
        
        # Récupération des paramètres de marché depuis la configuration
        self.market_params = config['market']
        self.ticker = self.market_params['ticker']
        self.reference_date = self.market_params['reference_date']
        self.r = self.market_params['risk_free_rate']
        self.q = self.market_params['dividend_yield']
        
        # Initialisation des structures de données pour les résultats
        self.spot_price = None
        self.processed_data = {
            OPTION_TYPES['CALL']: [],
            OPTION_TYPES['PUT']: []
        }
        
        self.logger.info(f"Analyseur initialisé pour {self.ticker}")

    def process_expiration(self, expiration: str) -> bool:
        """
        Traite les données pour une date d'expiration spécifique.
        
        Args:
            expiration: Date d'expiration au format 'YYYY-MM-DD'
            
        Returns:
            bool: True si le traitement a réussi, False sinon
        """
        try:
            # Calcul du temps jusqu'à maturité
            expiry_date = datetime.datetime.strptime(expiration, "%Y-%m-%d").date()
            T = (expiry_date - self.reference_date).days / 365.0
            
            if T <= 0:
                self.logger.warning(f"Expiration {expiration} déjà passée")
                return False

            # Récupération des données de marché
            S, calls_df, puts_df = fetch_option_chain(
                self.ticker,
                expiration,
                cache_dir=config['cache']['cache_dir']
            )
            
            if S is None or calls_df is None or puts_df is None:
                self.logger.error(f"Données invalides pour {expiration}")
                return False
            
            self.spot_price = S
            processor = OptionDataProcessor(S, self.r, self.q)
            
            # Traitement des calls et puts
            for opt_type, df in [(OPTION_TYPES['CALL'], calls_df),
                               (OPTION_TYPES['PUT'], puts_df)]:
                processed = processor.process_chain(df, T, opt_type)
                if processed is not None:
                    self.processed_data[opt_type].append(processed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur pour {expiration}: {str(e)}")
            return False

    def analyze_volatility_structure(self):
        """Analyse la structure de volatilité pour détecter les arbitrages."""
        for opt_type in OPTION_TYPES.values():
            data_list = self.processed_data[opt_type]
            
            if not data_list:
                continue
                
            # Vérification des arbitrages intra-maturity
            for df in data_list:
                if not OptionDataProcessor.check_no_arbitrage(
                    df, opt_type, config['arbitrage']['price_epsilon']
                ):
                    self.logger.warning(
                        f"Arbitrage détecté pour {opt_type}, T={df['T'].iloc[0]:.2f}"
                    )
            
            # Vérification des arbitrages calendaires
            if not OptionDataProcessor.check_calendar_arbitrage(data_list):
                self.logger.warning(
                    f"Arbitrage calendaire possible pour {opt_type}"
                )

    def create_visualizations(self):
        """Génère les visualisations des surfaces de volatilité."""
        for opt_type, data_list in self.processed_data.items():
            if not data_list:
                continue

            # Surface 3D complète
            all_data = pd.concat(data_list, ignore_index=True)
            plot_volatility_surface(
                all_data,
                self.spot_price,
                opt_type,
                plot_type='surface',
                model_col='implied_vol_BS',
                use_moneyness=config['plot']['use_moneyness']
            )
            
            # Comparaison des smiles
            plot_volatility_surface(
                data_list,
                self.spot_price,
                opt_type,
                plot_type='comparison',
                model_col='implied_vol_BS',
                dates=self.market_params['expirations'],
                use_moneyness=config['plot']['use_moneyness']
            )

    def calibrate_models(self):
        """Calibre et compare les modèles de Heston."""
        if not any(self.processed_data.values()):
            self.logger.warning("Données insuffisantes pour la calibration")
            return

        # Combine toutes les données
        all_data = pd.concat(
            sum(self.processed_data.values(), []),
            ignore_index=True
        )
        
        # Calibration sur volatilités BS
        heston_bs, _ = calibrate_heston_model_from_iv(
            self.reference_date,
            self.spot_price,
            self.r,
            self.q,
            all_data,
            "implied_vol_BS"
        )
        
        # Calibration sur volatilités CRR
        heston_crr, _ = calibrate_heston_model_from_iv(
            self.reference_date,
            self.spot_price,
            self.r,
            self.q,
            all_data,
            "implied_vol_CRR"
        )
        
        if heston_bs and heston_crr:
            self._compare_model_prices(heston_bs, heston_crr)

    def _compare_model_prices(self, heston_bs, heston_crr):
        """Compare les prix entre les différentes calibrations Heston."""
        # Points de test représentatifs
        test_points = [
            (0.8, 0.5, "ITM"),   # In-the-money
            (1.0, 0.5, "ATM"),   # At-the-money
            (1.2, 0.5, "OTM"),   # Out-of-the-money
            (1.0, 1.0, "ATM Long")
        ]
        
        self.logger.info("\nComparaison des prix Heston:")
        header = f"{'Moneyness':^10} {'Maturité':^10} {'Type':^8} "
        header += f"{'Call (BS)':^12} {'Call (CRR)':^12} {'Diff %':^8}"
        self.logger.info("\n" + header)
        self.logger.info("-" * len(header))
        
        for moneyness, T, desc in test_points:
            K = self.spot_price * moneyness
            
            price_bs = price_european_option_heston(heston_bs, K, T, "C")
            price_crr = price_european_option_heston(heston_crr, K, T, "C")
            
            if price_bs and price_crr:
                diff_pct = 100 * abs(price_bs - price_crr) / price_bs
                self.logger.info(
                    f"{moneyness:^10.2f} {T:^10.2f} {desc:^8s} "
                    f"{price_bs:^12.4f} {price_crr:^12.4f} {diff_pct:^8.2f}"
                )

    def run_full_analysis(self) -> bool:
        """
        Exécute l'analyse complète des options.
        
        Returns:
            bool: True si l'analyse s'est bien déroulée
        """
        try:
            self.logger.info(f"\nDébut de l'analyse pour {self.ticker}")
            self.logger.info(f"Date de référence: {self.reference_date}\n")
            
            # 1. Traitement des données par expiration
            for expiration in self.market_params['expirations']:
                self.logger.info(f"Traitement de {expiration}...")
                if not self.process_expiration(expiration):
                    self.logger.error(f"Échec pour {expiration}")
            
            # 2. Analyse de la structure de volatilité
            self.analyze_volatility_structure()
            
            # 3. Création des visualisations
            self.create_visualizations()
            
            # 4. Calibration des modèles (si activée)
            if config.get('calibrate_heston', False):
                self.calibrate_models()
            
            self.logger.info("\nAnalyse terminée avec succès")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse: {str(e)}")
            return False


def main():
    """Point d'entrée principal du programme."""
    analyzer = OptionAnalyzer()
    success = analyzer.run_full_analysis()
    if not success:
        logging.error("L'analyse a échoué")

if __name__ == "__main__":
    main()
