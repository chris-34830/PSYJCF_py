"""
Module de traitement des données d'options financières.

Ce module fournit les fonctionnalités nécessaires pour traiter et analyser
les données d'options, notamment :
- Le pré-traitement et le nettoyage des données brutes
- Le calcul des volatilités implicites selon différents modèles
- La détection d'opportunités d'arbitrage
- La validation des données selon divers critères de qualité

Le flux de traitement typique est le suivant :
1. Pré-traitement des données brutes (nettoyage, calcul de la moneyness)
2. Application des filtres de qualité
3. Calcul des volatilités implicites
4. Vérification des conditions d'absence d'arbitrage
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Union

from ..config import config
from ..models.implied_vol import implied_volatility
from ..utils.market_utils import prepare_option_data, get_clean_market_price
from ..constants import OPTION_TYPES, COLUMN_NAMES

class OptionDataProcessor:
    """
    Classe principale pour le traitement des données d'options.
    
    Cette classe encapsule toute la logique nécessaire pour traiter
    les données d'options, de leur état brut jusqu'aux volatilités
    implicites et aux vérifications d'arbitrage.
    """
    
    def __init__(self, spot_price: float, risk_free_rate: float, 
                 dividend_yield: float):
        """
        Initialise le processeur de données d'options.
        
        Args:
            spot_price (float): Prix spot de l'actif sous-jacent
            risk_free_rate (float): Taux sans risque annualisé
            dividend_yield (float): Taux de dividende annualisé
        """
        self.S = spot_price
        self.r = risk_free_rate
        self.q = dividend_yield
        self.logger = logging.getLogger(__name__)

    def process_chain(self, 
                     raw_data: pd.DataFrame,
                     time_to_maturity: float,
                     option_type: str) -> Optional[pd.DataFrame]:
        """
        Traite une chaîne d'options complète.
        
        Cette méthode coordonne le processus complet de traitement :
        1. Pré-traitement et nettoyage des données
        2. Calcul des prix de marché et application des filtres
        3. Calcul des volatilités implicites pour tous les modèles
        
        Args:
            raw_data (pd.DataFrame): Données brutes de la chaîne d'options
            time_to_maturity (float): Temps jusqu'à l'expiration en années
            option_type (str): Type d'option ('C' pour call, 'P' pour put)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame traité ou None en cas d'erreur
        """
        try:
            # 1. Pré-traitement initial
            df = self._preprocess_data(raw_data, time_to_maturity, option_type)
            if df is None:
                return None
                
            # 2. Calcul des prix de marché et filtrage
            df = self._apply_market_filters(df)
            
            # 3. Calcul des volatilités implicites
            df = self._compute_implied_volatilities(df, option_type)
            
            self.logger.info(
                f"Traitement réussi pour {option_type} - T={time_to_maturity:.3f}"
                f" ({len(df[df['included']])} options valides sur {len(df)})"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(
                f"Erreur lors du traitement de la chaîne {option_type}: {str(e)}"
            )
            return None
            
    def _preprocess_data(self,
                        df: pd.DataFrame,
                        T: float,
                        option_type: str) -> Optional[pd.DataFrame]:
        """
        Effectue le pré-traitement initial des données.
        
        Cette étape comprend :
        - Le nettoyage des données manquantes
        - Le calcul de la moneyness
        - L'ajout des métadonnées (type d'option, maturité)
        - La préparation des colonnes pour la suite du traitement
        
        Args:
            df (pd.DataFrame): Données brutes
            T (float): Temps jusqu'à maturité
            option_type (str): Type d'option
            
        Returns:
            Optional[pd.DataFrame]: Données pré-traitées
        """
        if df is None or df.empty:
            self.logger.warning("Données d'entrée vides ou invalides")
            return None
            
        try:
            # Copie pour éviter la modification des données d'origine
            df = df.copy()
            
            # Ajout des métadonnées
            df['T'] = T
            df['option_type'] = option_type
            
            # Préparation des données de marché
            df = prepare_option_data(
                df, 
                self.S,
                option_type,
                use_moneyness=config['plot']['use_moneyness']
            )
            
            # Nettoyage des données invalides
            df = df.dropna(subset=['strike', 'bid', 'ask'])
            df = df[df['strike'] > 0]
            
            # Initialisation des colonnes de filtrage
            df['included'] = False
            df['exclusion_reason'] = ""
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors du pré-traitement: {str(e)}")
            return None

    def _apply_market_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique les filtres de qualité aux données de marché.
        
        Les filtres incluent :
        - Validité des prix bid/ask
        - Spread bid/ask raisonnable
        - Volume et open interest minimaux
        - Distance raisonnable du strike par rapport au spot
        
        Args:
            df (pd.DataFrame): Données à filtrer
            
        Returns:
            pd.DataFrame: Données avec les filtres appliqués
        """
        for idx, row in df.iterrows():
            failed_filters = self._check_filters(row)
            
            if failed_filters:
                df.at[idx, 'included'] = False
                df.at[idx, 'exclusion_reason'] = '; '.join(failed_filters)
            else:
                df.at[idx, 'included'] = True
                df.at[idx, 'exclusion_reason'] = ""
                
        return df

    def _check_filters(self, row: pd.Series) -> List[str]:
        """
        Vérifie tous les critères de qualité pour une option.
        
        Args:
            row (pd.Series): Ligne de données pour une option
            
        Returns:
            List[str]: Liste des raisons d'exclusion (vide si tout est valide)
        """
        failed_filters = []
        
        # 1. Vérification bid/ask
        bid, ask = row.get('bid', np.nan), row.get('ask', np.nan)
        if pd.isna(bid) or pd.isna(ask) or bid < 0 or ask <= 0 or ask < bid:
            failed_filters.append("Bid/Ask invalide")
        else:
            # Vérification du spread
            mid = 0.5 * (bid + ask)
            spread = ask - bid
            if mid <= 0:
                failed_filters.append("Point médian <= 0")
            elif spread/mid > config['filters']['max_spread_ratio']:
                failed_filters.append("Spread excessif")
                
        # 2. Vérification volume/open interest
        vol = row.get('volume', 0)
        oi = row.get('openInterest', 0)
        if vol < config['filters']['min_volume']:
            failed_filters.append("Volume insuffisant")
        if oi < config['filters']['min_open_interest']:
            failed_filters.append("Open interest insuffisant")
            
        # 3. Vérification de la distance du strike
        strike = row['strike']
        if not (config['filters']['strike_min_factor'] * self.S <= strike <= 
                config['filters']['strike_max_factor'] * self.S):
            failed_filters.append("Strike trop éloigné")
            
        return failed_filters

    def _compute_implied_volatilities(self,
                                    df: pd.DataFrame,
                                    option_type: str) -> pd.DataFrame:
        """
        Calcule les volatilités implicites selon différents modèles.
        
        Calcule les volatilités implicites pour :
        - Le modèle de Black-Scholes (BS)
        - Le modèle binomial (CRR)
        - Le modèle de Barone-Adesi-Whaley (BAW)
        
        Args:
            df (pd.DataFrame): Données d'options filtrées
            option_type (str): Type d'option ('C' ou 'P')
            
        Returns:
            pd.DataFrame: Données avec les volatilités implicites calculées
        """
        for idx, row in df.iterrows():
            if not row['included']:
                continue
                
            T = row['T']
            K = row['strike']
            price = row['market_price']
            
            # Ajustement du nombre de pas CRR selon la moneyness
            steps = config['model']['crr_default_steps']
            if ((option_type == 'C' and K < 0.8 * self.S) or
                (option_type == 'P' and K > 1.2 * self.S)):
                steps = config['model']['crr_reduced_steps']
            
            # Calcul des volatilités pour chaque modèle
            try:
                df.at[idx, 'implied_vol_CRR'] = implied_volatility(
                    price, self.S, K, self.r, self.q, T, 'CRR', 
                    option_type, steps
                )
                df.at[idx, 'implied_vol_BS'] = implied_volatility(
                    price, self.S, K, self.r, self.q, T, 'BS', 
                    option_type
                )
                df.at[idx, 'implied_vol_BAW'] = implied_volatility(
                    price, self.S, K, self.r, self.q, T, 'BAW', 
                    option_type
                )
            except Exception as e:
                self.logger.warning(
                    f"Échec du calcul des volatilités pour K={K}: {str(e)}"
                )
                df.at[idx, 'included'] = False
                df.at[idx, 'exclusion_reason'] = "Échec calcul volatilités"
                
        return df

    @staticmethod
    def check_no_arbitrage(df: pd.DataFrame,
                          option_type: str,
                          epsilon: float = None) -> bool:
        """
        Vérifie l'absence d'opportunités d'arbitrage.
        
        Vérifie deux types d'arbitrage :
        1. Arbitrage de spread vertical (monotonie des prix)
        2. Arbitrage papillon (convexité des prix)
        
        Args:
            df (pd.DataFrame): Données d'options
            option_type (str): Type d'option ('C' ou 'P')
            epsilon (float, optional): Tolérance pour les violations d'arbitrage
            
        Returns:
            bool: True si aucun arbitrage n'est détecté
        """
        if epsilon is None:
            epsilon = config['arbitrage']['price_epsilon']
            
        price_col = COLUMN_NAMES['price'][option_type]
        df_clean = df.dropna(subset=['strike', price_col]).sort_values('strike')
        
        if df_clean.empty:
            return True
            
        prices = df_clean[price_col].values
        strikes = df_clean['strike'].values
        
        # 1. Vérification de la monotonie
        for i in range(len(prices) - 1):
            if option_type == 'C':
                # Prix des calls doivent décroître avec le strike
                if prices[i+1] > prices[i] + epsilon:
                    logging.warning(
                        f"Violation monotonie CALL: K={strikes[i]:.1f}-"
                        f"{strikes[i+1]:.1f}, P={prices[i]:.2f}-{prices[i+1]:.2f}"
                    )
                    return False
            else:
                # Prix des puts doivent croître avec le strike
                if prices[i+1] < prices[i] - epsilon:
                    logging.warning(
                        f"Violation monotonie PUT: K={strikes[i]:.1f}-"
                        f"{strikes[i+1]:.1f}, P={prices[i]:.2f}-{prices[i+1]:.2f}"
                    )
                    return False
        
        # 2. Vérification de la convexité (arbitrage papillon)
        for i in range(1, len(prices) - 1):
            # La fonction de prix doit être convexe : 2*P(K2) ≤ P(K1) + P(K3)
            butterfly = 2.0 * prices[i] - (prices[i-1] + prices[i+1])
            if butterfly > epsilon:
                logging.warning(
                    f"Violation convexité: K={strikes[i-1]:.1f}-{strikes[i]:.1f}-"
                    f"{strikes[i+1]:.1f}, P={prices[i-1]:.2f}-{prices[i]:.2f}-"
                    f"{prices[i+1]:.2f}"
                )
                return False
                
        return True

    @staticmethod
    def check_calendar_arbitrage(dfs: List[pd.DataFrame],
                               model_col: str = "implied_vol_BS") -> bool:
        """
        Vérifie l'absence d'arbitrage calendaire.
        
        La variance totale (σ²T) doit être croissante avec la maturité
        pour éviter les opportunités d'arbitrage calendaire.
        
        Args:
            dfs (List[pd.DataFrame]): Liste des DataFrames par maturité
            model_col (str): Colonne contenant les volatilités à vérifier
            
        Returns:
            bool: True si aucun arbitrage calendaire n'est détecté
        """
        if not dfs:
            return True
            
        # Tri par maturité croissante
        sorted_dfs = sorted(dfs, key=lambda df: df['T'].iloc[0])
        last_total_variance = -1.0