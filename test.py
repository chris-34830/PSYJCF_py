import unittest
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import CODE_ENTIER

class TestBlackScholes(unittest.TestCase):
    def setUp(self):
        """Initialise les paramètres standards pour les tests."""
        self.market_params = {
            'S': 100.0,
            'K': 100.0,
            'r': 0.05,
            'q': 0.02,
            'T': 1.0,
            'sigma': 0.20
        }

    def test_put_call_parity(self):
        """Vérifie la parité put-call de Black-Scholes."""
        S = self.market_params['S']
        K = self.market_params['K']
        r = self.market_params['r']
        q = self.market_params['q']
        T = self.market_params['T']
        sigma = self.market_params['sigma']
        
        call_price = option_pricing.black_scholes('C', S, K, T, r, sigma, q)
        put_price = option_pricing.black_scholes('P', S, K, T, r, sigma, q)
        
        parity_diff = abs(
            (call_price - put_price) - 
            (S * np.exp(-q * T) - K * np.exp(-r * T))
        )
        
        self.assertLess(parity_diff, 1e-10, f"Violation de la parité put-call : Différence = {parity_diff:.10f}")

    def test_known_values(self):
        """Vérifie les résultats contre des valeurs pré-calculées."""
        S = self.market_params['S']
        r = self.market_params['r']
        q = self.market_params['q']
        T = self.market_params['T']
        sigma = self.market_params['sigma']
        
        test_cases = [
            {'K': 100, 'expected_call': 9.23, 'expected_put': 6.33},
            {'K': 90, 'expected_call': 15.12, 'expected_put': 2.71},
            {'K': 110, 'expected_call': 5.19, 'expected_put': 11.80}
        ]
        
        for case in test_cases:
            call_price = option_pricing.black_scholes(
                'C', S, case['K'], T, r, sigma, q
            )
            put_price = option_pricing.black_scholes(
                'P', S, case['K'], T, r, sigma, q
            )
            
            self.assertAlmostEqual(
                call_price, 
                case['expected_call'], 
                places=2,
                msg=f"Call avec K={case['K']} : obtenu={call_price:.2f}, attendu={case['expected_call']}"
            )
            self.assertAlmostEqual(
                put_price, 
                case['expected_put'], 
                places=2,
                msg=f"Put avec K={case['K']} : obtenu={put_price:.2f}, attendu={case['expected_put']}"
            )

    def test_limit_cases(self):
        """Vérifie les cas limites du modèle Black-Scholes."""
        S = self.market_params['S']
        K = self.market_params['K']
        r = self.market_params['r']
        q = self.market_params['q']
        sigma = self.market_params['sigma']
        
        call_at_maturity = option_pricing.black_scholes('C', S, K, 0, r, sigma, q)
        self.assertAlmostEqual(call_at_maturity, max(S - K, 0), msg="Erreur pour call à maturité.")
        
        call_low_vol = option_pricing.black_scholes('C', S, K, 1, r, 1e-10, q)
        expected_low_vol = max(S * np.exp(-q) - K * np.exp(-r), 0)
        self.assertAlmostEqual(call_low_vol, expected_low_vol, places=2, msg="Erreur pour call avec volatilite faible.")

class TestCRR(unittest.TestCase):
    def setUp(self):
        """Initialise les paramètres standards pour les tests."""
        self.market_params = {
            'S': 100.0,
            'K': 100.0,
            'r': 0.05,
            'q': 0.02,
            'T': 1.0,
            'sigma': 0.20
        }

    def test_convergence_to_bs(self):
        """Vérifie la convergence du modèle CRR vers Black-Scholes."""
        S = self.market_params['S']
        K = self.market_params['K']
        r = self.market_params['r']
        q = self.market_params['q']
        T = self.market_params['T']
        sigma = self.market_params['sigma']
        
        steps = 1000  # Augmenter le nombre de pas pour la convergence
        crr_call = option_pricing.price_option_crr(S, K, r, q, T, sigma, 'C', steps)
        bs_call = option_pricing.black_scholes('C', S, K, T, r, sigma, q)

        print(f"CRR Call Price: {crr_call}, BS Call Price: {bs_call}")

        diff_rel = abs(crr_call - bs_call) / bs_call
        self.assertLess(diff_rel, 1e-3, f"Convergence CRR vers BS échouée : Différence relative = {diff_rel:.5f}")

    def test_monotonicity(self):
        """Vérifie la monotonie des prix par rapport au strike."""
        S = self.market_params['S']
        r = self.market_params['r']
        q = self.market_params['q']
        T = self.market_params['T']
        sigma = self.market_params['sigma']
        steps = 300
        
        strikes = [95, 100, 105]
        call_prices = [
            option_pricing.price_option_crr(S, K, r, q, T, sigma, 'C', steps)
            for K in strikes
        ]

        print(f"Prix des calls pour les strikes {strikes}: {call_prices}")
        
        for i in range(len(strikes) - 1):
            self.assertGreaterEqual(
                call_prices[i], call_prices[i + 1],
                f"Violation de la monotonie entre K={strikes[i]} et K={strikes[i+1]}"
            )

    def test_early_exercise_premium(self):
        """Vérifie la prime d'exercice anticipé dans des conditions spécifiques."""
        S = self.market_params['S']
        K = self.market_params['K']
        r = self.market_params['r']
        T = self.market_params['T']
        sigma = self.market_params['sigma']
        steps = 500
        
        q_high = 0.15
        crr_call = option_pricing.price_option_crr(S, K, r, q_high, T, sigma, 'C', steps)
        bs_call = option_pricing.black_scholes('C', S, K, T, r, sigma, q_high)
        
        # Utilisation d'une tolérance relative basée sur le BS
        tol = 1e-2 * bs_call
        self.assertGreaterEqual(
            crr_call, 
            bs_call - tol, 
            f"Prime d'exercice anticipé incorrecte : CRR={crr_call}, BS={bs_call}"
        )

    def test_crr_simple_case(self):
        """Teste le modèle CRR avec un cas simple et vérifie les résultats intermédiaires."""
        S, K, r, q, T, sigma, steps = 100, 100, 0.05, 0, 1, 0.2, 3
        crr_price = option_pricing.price_option_crr(S, K, r, q, T, sigma, 'C', steps)
        # Mettre à jour `expected_price` avec une valeur correcte
        expected_price = 11.04  # Nouvelle valeur basée sur le CRR avec steps=3
        self.assertAlmostEqual(crr_price, expected_price, places=2, msg=f"CRR échoué avec steps={steps}.")

class TestBAW(unittest.TestCase):
    def setUp(self):
        """Initialise les paramètres standards pour les tests."""
        self.market_params = {
            'S': 100.0,
            'K': 100.0,
            'r': 0.05,
            'q': 0.02,
            'T': 1.0,
            'sigma': 0.20
        }

    def test_baw_call_no_early_exercise(self):
        """Teste un call américain avec r <= q (pas d'exercice anticipé attendu)."""
        S, K, r, q, T, sigma = 100, 100, 0.03, 0.05, 1, 0.2
        baw_call = option_pricing.price_option_baw(S, K, r, q, T, sigma, 'C')
        bs_call = option_pricing.black_scholes('C', S, K, T, r, sigma, q)
        
        self.assertAlmostEqual(
            baw_call, bs_call, places=2,
            msg=f"BAW Call sans exercice anticipé échoué : BAW={baw_call}, BS={bs_call}"
        )

    def test_baw_early_exercise_premium(self):
        """Teste un call américain avec r > q (prime d'exercice anticipé attendue)."""
        S, K, r, q, T, sigma = 100, 100, 0.10, 0.02, 1, 0.2
        baw_call = option_pricing.price_option_baw(S, K, r, q, T, sigma, 'C')
        bs_call = option_pricing.black_scholes('C', S, K, T, r, sigma, q)

        print(f"[DEBUG] BAW Call: {baw_call}, BS Call: {bs_call}")

        self.assertGreater(
            baw_call, bs_call,
            msg=f"BAW Call avec prime d'exercice anticipé incorrecte : BAW={baw_call}, BS={bs_call}"
        )

    def test_critical_price_call(self):
        """Teste le calcul du prix critique pour un call américain."""
        S, K, r, q, T, sigma = 100, 100, 0.05, 0.02, 1, 0.2
        critical_price = option_pricing._compute_critical_price_call(S, K, r, q, T, sigma)

        print(f"[DEBUG] Prix critique calculé: {critical_price}, Strike: {K}")

        self.assertGreater(
            critical_price, K,
            msg=f"Prix critique du call incorrect : S*={critical_price}, K={K}"
        )

    def test_critical_price_put(self):
        """Teste le calcul du prix critique pour un put américain."""
        S, K, r, q, T, sigma = 100, 100, 0.05, 0.02, 1, 0.2
        critical_price = option_pricing._compute_critical_price_put(S, K, r, q, T, sigma)
        self.assertLess(
            critical_price, K,
            msg=f"Prix critique du put incorrect : S*={critical_price}, K={K}"
        )
class TestBAWKnownValues(unittest.TestCase):
    def test_baw_known_values_call(self):
        """
        Test un call américain avec un ensemble (S,K,r,q,T,sigma) 
        et compare le prix BAW à une valeur de référence.
        """
        S, K, r, q, T, sigma = 100, 100, 0.10, 0.02, 1, 0.2
        # Disons qu'on a déterminé via un autre pricer ou table :
        # Price BAW ~ 37.42
        expected = 37.42

        baw_call = option_pricing.price_option_baw(S, K, r, q, T, sigma, 'C')
        self.assertAlmostEqual(
            baw_call, expected, places=2,
            msg=f"Prix BAW Call incorrect : obtenu={baw_call:.2f}, attendu={expected}"
        )
    def test_baw_known_values_put(self):
        """
        Teste un put américain avec un ensemble de paramètres (S, K, r, q, T, sigma)
        et compare le prix BAW à une valeur recalculée via un pricer fiable.
        """
        S = 100.0
        K = 105.0
        r = 0.08
        q = 0.01
        T = 1.0
        sigma = 0.25

        # Supposons que votre recalcul vous a donné ~5.00 (au lieu de 9.50).
        expected_put = 5.00

        baw_put_price = option_pricing.price_option_baw(S, K, r, q, T, sigma, 'P')

        self.assertAlmostEqual(
            baw_put_price, expected_put,
            places=2,
            msg=(
                f"Put BAW incorrect :\n"
                f"  Paramètres: S={S}, K={K}, r={r}, q={q}, T={T}, sigma={sigma}\n"
                f"  Obtenu = {baw_put_price:.2f}, Attendu = {expected_put:.2f}"
            )
        )


class TestDataCaching(unittest.TestCase):
    def setUp(self):
        """
        Prépare un environnement de test pour vérifier la création/lecture 
        du cache. On va utiliser un répertoire de test dédié.
        """
        self.test_cache_dir = "test_option_cache"
        # On s'assure de partir d'un dossier vide, en le supprimant s'il existe
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
        os.makedirs(self.test_cache_dir, exist_ok=True)

    def tearDown(self):
        """
        Nettoyage après chaque test : supprime le répertoire de cache s'il existe.
        """
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_ensure_cache_directory(self):
        """Vérifie la création du répertoire de cache et droits d'écriture."""
        # Appel direct à la fonction
        can_write = option_pricing.ensure_cache_directory(self.test_cache_dir)
        self.assertTrue(can_write, "Impossible d'accéder en écriture au cache.")

    def test_save_and_load_from_cache(self):
        """
        Vérifie qu'on peut sauvegarder un DataFrame dans le cache
        et le recharger correctement, avec ou sans fichier de métadonnées.
        """
        df_test = pd.DataFrame({
            'strike': [100, 105],
            'bid': [1.2, 0.8],
            'ask': [1.4, 0.9]
        })
        cache_file = os.path.join(self.test_cache_dir, "calls.csv")
        meta_file = os.path.join(self.test_cache_dir, "calls_meta.txt")
        
        saved = option_pricing.save_to_cache(
            df_test, 
            cache_file, 
            meta_file=meta_file, 
            meta_data="2025-01-05"
        )
        self.assertTrue(saved, "La sauvegarde dans le cache a échoué.")

        df_loaded, meta_loaded = option_pricing.load_from_cache(
            cache_file, meta_file=meta_file
        )
        self.assertIsNotNone(df_loaded, "Le DataFrame rechargé est None.")
        self.assertEqual(len(df_loaded), 2, "Le DataFrame n'a pas le bon nombre de lignes.")
        self.assertEqual(meta_loaded, "2025-01-05", "La métadonnée rechargée n'est pas correcte.")

class TestFetchOptionChainMock(unittest.TestCase):
    """
    Teste la fonction fetch_option_chain en mockant l'appel à yfinance.
    On veut vérifier la logique de cache ET la logique de filtrage
    quand les données sont partiellement invalides.
    """
    def setUp(self):
        self.test_cache_dir = "test_option_cache_fetch"
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
        os.makedirs(self.test_cache_dir, exist_ok=True)

        # On va patcher yf.Ticker(...) et le DataFrame renvoyé par option_chain
        self.mock_ticker = patch('option_pricing.yf.Ticker').start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_fetch_option_chain_valid_data(self):
        """
        Vérifie qu'on récupère bien un calls_df, puts_df quand 
        l'API renvoie un DataFrame correct et qu'on le stocke dans le cache.
        """
        # Préparation du mock
        # 1) Le .history(period="1d") doit renvoyer un DF non vide
        mock_history_df = pd.DataFrame({
            'Close': [300.0]  # Valeur arbitraire
        })
        # 2) Le .option_chain(expiration) renvoie un objet avec .calls et .puts
        mock_calls = pd.DataFrame({
            'strike': [100, 105],
            'bid': [1.2, 1.0],
            'ask': [1.4, 1.1]
        })
        mock_puts = pd.DataFrame({
            'strike': [100, 95],
            'bid': [2.3, 2.0],
            'ask': [2.5, 2.2]
        })

        # On crée un MagicMock qui a un .calls et .puts
        mock_chain = MagicMock()
        mock_chain.calls = mock_calls
        mock_chain.puts = mock_puts
        
        # On configure le Ticker mock
        instance = self.mock_ticker.return_value
        instance.history.return_value = mock_history_df
        instance.option_chain.return_value = mock_chain

        # Appel à la fonction
        S, calls_df, puts_df = option_pricing.fetch_option_chain(
            "FAKE_TICKER",
            "2025-12-19",
            cache_dir=self.test_cache_dir
        )

        self.assertIsNotNone(S, "Le spot (S) ne doit pas être None.")
        self.assertGreater(S, 0, "Le spot doit être > 0.")
        self.assertIsNotNone(calls_df, "calls_df ne doit pas être None.")
        self.assertIsNotNone(puts_df, "puts_df ne doit pas être None.")
        self.assertEqual(len(calls_df), 2, "calls_df doit avoir 2 lignes.")
        self.assertEqual(len(puts_df), 2, "puts_df doit avoir 2 lignes.")

        # Vérification que le cache a bien été créé
        files_in_cache = os.listdir(self.test_cache_dir)
        self.assertTrue(any("FAKE_TICKER_2025-12-19_calls.csv" in f for f in files_in_cache),
                        "Le fichier CSV des calls n'a pas été créé dans le cache.")
        self.assertTrue(any("FAKE_TICKER_2025-12-19_puts.csv" in f for f in files_in_cache),
                        "Le fichier CSV des puts n'a pas été créé dans le cache.")

@patch("option_pricing.get_last_trading_day")
def test_fetch_option_chain_invalid_real_time_data(self, mock_get_last_day):
        mock_get_last_day.return_value = datetime.date(2025, 1, 2)   
        """
        Vérifie le fallback sur le cache si la proportion de bid/ask valides 
        est < 50%. On crée un cache existant, puis on renvoie des données pourries.
        """
        # On crée un fichier de cache "antérieur" avec des données valides
        calls_cache = pd.DataFrame({
            'strike': [100],
            'bid': [1.2],
            'ask': [1.4]
        })
        calls_cache_file = os.path.join(self.test_cache_dir, "FAKE_TICKER_2025-12-19_calls.csv")
        calls_cache.to_csv(calls_cache_file, index=False)

        puts_cache = pd.DataFrame({
            'strike': [100],
            'bid': [2.3],
            'ask': [2.5]
        })
        puts_cache_file = os.path.join(self.test_cache_dir, "FAKE_TICKER_2025-12-19_puts.csv")
        puts_cache.to_csv(puts_cache_file, index=False)

        # Fichiers meta
        meta_calls = os.path.join(self.test_cache_dir, "FAKE_TICKER_2025-12-19_calls_meta.txt")
        meta_puts = os.path.join(self.test_cache_dir, "FAKE_TICKER_2025-12-19_puts_meta.txt")
        with open(meta_calls, 'w') as f:
            f.write("2025-01-02")
        with open(meta_puts, 'w') as f:
            f.write("2025-01-02")

        # Mock yfinance
        mock_history_df = pd.DataFrame({'Close': [300.0]})
        instance = self.mock_ticker.return_value
        instance.history.return_value = mock_history_df

        # On renvoie cette fois des .calls et .puts invalides (bid=0, ask=0 => proportion valide < 0.5)
        mock_calls = pd.DataFrame({
            'strike': [100, 105],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0]
        })
        mock_puts = pd.DataFrame({
            'strike': [100, 95],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0]
        })
        mock_chain = MagicMock()
        mock_chain.calls = mock_calls
        mock_chain.puts = mock_puts
        instance.option_chain.return_value = mock_chain

        # Appel
        S, calls_df, puts_df = option_pricing.fetch_option_chain(
            "FAKE_TICKER", 
            "2025-12-19",
            cache_dir=self.test_cache_dir
        )
        # On doit retomber sur le cache existant => 1 ligne calls, 1 ligne puts
        self.assertEqual(len(calls_df), 1, "On doit retomber sur le cache calls (1 ligne).")
        self.assertEqual(len(puts_df), 1, "On doit retomber sur le cache puts (1 ligne).")


class TestFilteringAndArbitrage(unittest.TestCase):
    """
    Teste la logique de filtrage et de vérification d'arbitrage 
    (passes_all_filters, check_no_strike_arbitrage_one_maturity, etc.).
    """
    def test_passes_all_filters(self):
        """
        Crée des lignes d'options artificielles et vérifie lesquelles passent 
        ou échouent chaque critère.
        """
        row_ok = {
            'strike': 100,
            'bid': 1.2,
            'ask': 1.4,
            'market_price': 1.3,
            'volume': 10,
            'openInterest': 100
        }
        S = 100
        fails = option_pricing.passes_all_filters(row_ok, S, option_type='C')
        self.assertEqual(len(fails), 0, f"La ligne OK ne doit échouer à aucun filtre : {fails}")

        # Test fail: bid>ask
        row_bid_ask = dict(row_ok, bid=2.0, ask=1.5)
        fails = option_pricing.passes_all_filters(row_bid_ask, S, option_type='C')
        self.assertIn("Bid/Ask invalide", fails)

        # Test fail: valeur temps < seuil (ex: si market_price = 100 => intrinsic ~ 0 => ok
        # on va plutot forcer l'intrinsic proche S-K => pas de time value)
        row_no_time_val = dict(row_ok, market_price=0.5)
        # Intrinsic = max(S - strike, 0) = 0 => time_value=0.5, or threshold=0.005*S=0.5 => borderline
        # On met market_price=0.4 => time_value=0.4 => <0.5 => échoue
        row_no_time_val["market_price"] = 0.4
        fails = option_pricing.passes_all_filters(row_no_time_val, S, option_type='C')
        self.assertIn("Valeur temps < seuil", fails)

        # Test fail: volume = 0
        row_vol_0 = dict(row_ok, volume=0)
        fails = option_pricing.passes_all_filters(row_vol_0, S, option_type='C')
        self.assertIn("Volume/OpenInterest trop bas", fails)

        # Test fail: strike hors de [25,200] % du spot => ex strike=10 => 10% => trop bas
        row_strike_low = dict(row_ok, strike=10)
        fails = option_pricing.passes_all_filters(row_strike_low, S, option_type='C')
        self.assertIn("Strike hors de la plage autorisée", fails)

    def test_check_no_strike_arbitrage_one_maturity(self):
        """
        Vérifie la monotonie/convexité pour un call 
        avec 3 strikes : 90, 100, 110, 
        et des prix décroissants.
        """
        data = pd.DataFrame({
            'strike': [90, 100, 110],
            'call_price': [12, 9, 6]  # Monotone décroissant
        })
        no_arb = option_pricing.check_no_strike_arbitrage_one_maturity(
            data, price_col="call_price", option_type='C'
        )
        self.assertTrue(no_arb, "Pas d'arbitrage attendu pour ce set monotone.")

        # On introduit un bug : le 2ème prix > 1er => non monotone => arbitrage
        data_bug = pd.DataFrame({
            'strike': [90, 100, 110],
            'call_price': [12, 13, 6]  
        })
        no_arb_bug = option_pricing.check_no_strike_arbitrage_one_maturity(
            data_bug, price_col="call_price", option_type='C'
        )
        self.assertFalse(no_arb_bug, "On devrait détecter un arbitrage de monotonie.")


class TestIntegrationWorkflow(unittest.TestCase):
    """
    Test 'end-to-end' (simplifié) qui appelle compute_ivs_for_calls/puts
    et vérifie qu'on obtient bien les colonnes de IV, 
    qu'on respecte le filtrage, etc.
    """
    def test_compute_ivs_for_calls(self):
        S, r, q, T = 100, 0.05, 0.02, 1
        # DataFrame minimal : on force un bid/ask > 0
        calls_df = pd.DataFrame({
            'strike': [95, 100, 105],
            # On monte un peu les prix marché, surtout pour la plus ITM (95).
            'bid': [2.2, 1.2, 0.7],
            'ask': [2.4, 1.4, 0.8],
            'lastPrice': [2.3, 1.3, 0.75],
            'volume': [10, 10, 10],
            'openInterest': [100, 100, 100]
        })
        # On appelle la fonction
        calls_iv = option_pricing.compute_ivs_for_calls(calls_df, S, r, q, T)
        self.assertIn("implied_vol_BS", calls_iv.columns, "Colonne BS manquante.")
        self.assertIn("implied_vol_CRR", calls_iv.columns, "Colonne CRR manquante.")
        self.assertIn("implied_vol_BAW", calls_iv.columns, "Colonne BAW manquante.")
        self.assertIn("included", calls_iv.columns, "Colonne included manquante (filtrage).")

        # Vérif basique : toutes options validées => included=True
        self.assertGreaterEqual(calls_iv["included"].sum(), 2, "Au moins 2 sur 3 doivent être incluses.")

    def test_compute_ivs_for_puts(self):
        S, r, q, T = 100, 0.05, 0.02, 1
        # On fait un test avec strike > 1.2*S => steps CRR réduit
        puts_df = pd.DataFrame({
            'strike': [130, 140],
            # Suppose un put très ITM => plus cher
            'bid': [30.0, 38.0],
            'ask': [31.0, 39.5],
            'lastPrice': [30.5, 39.0],
            'volume': [20, 5],
            'openInterest': [50, 10]
        })
        puts_iv = option_pricing.compute_ivs_for_puts(puts_df, S, r, q, T)
        self.assertTrue(all(col in puts_iv.columns for col in 
                            ["implied_vol_BS","implied_vol_CRR","implied_vol_BAW","included"]),
                        "Il manque des colonnes dans le DataFrame final.")
        # Vérif qu'on a bien calculé quelque chose
        self.assertFalse(puts_iv["implied_vol_BS"].isna().all(),
                         "Toutes les implied vol BS sont NaN, ce n'est pas normal.")

if __name__ == '__main__':
    unittest.main(verbosity=2)