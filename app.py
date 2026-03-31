import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Simulateur de Rentabilité", page_icon="📈", layout="wide")

# ==========================================
# 1. CHARGEMENT DE L'IA
# ==========================================
@st.cache_resource
def charger_ia():
    try:
        modele = joblib.load('best_lightgbm_model.pkl')
        scaler = joblib.load('retail_scaler.pkl')
        return modele, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None

modele_lgbm, scaler = charger_ia()

# ==========================================
# BARRE LATÉRALE : PARAMÈTRES DÉTAILLÉS
# ==========================================
with st.sidebar:
    st.header("⚙️ Paramètres du Produit")
    
    st.markdown("**1. Vos Coûts**")
    cout_revient = st.number_input("Coût de revient unitaire (€)", min_value=1.0, value=25.0, step=0.5)
    frais_port = st.number_input("Frais de port (freight_price)", value=5.5)
    
    st.markdown("---")
    st.markdown("**2. La Concurrence**")
    comp_1 = st.number_input("Prix Concurrent 1 (€)", value=45.0)
    comp_2 = st.number_input("Prix Concurrent 2 (€)", value=49.9)
    comp_3 = st.number_input("Prix Concurrent 3 (€)", value=42.5)
    
    prix_moyen_marche = (comp_1 + comp_2 + comp_3) / 3
    
    st.markdown("---")
    st.markdown("**3. Historique & Qualité**")
    prix_precedent = st.number_input("Prix du mois dernier (lag_price)", value=48.0)
    score_produit = st.slider("Note client (1-5)", 1.0, 5.0, 4.0)
    nb_photos = st.number_input("Nombre de photos", min_value=1, value=3)
    nb_clients = st.number_input("Nombre de clients total", min_value=1, value=100)

    st.markdown("---")
    st.markdown("**4. Trafic & Visibilité (Nouveau 🚀)**")
    trafic_estime = st.number_input("Trafic estimé (Visiteurs ce mois-ci)", min_value=100, value=2500, step=100)
    
    st.markdown("---")
    mois_nom = st.selectbox("Mois de la simulation", ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"])

# ==========================================
# CORPS PRINCIPAL
# ==========================================
st.title("📈 Simulateur de Chiffre d'Affaires & Marge")
st.markdown("Testez votre stratégie de prix en fonction de votre trafic, de la concurrence et de votre historique.")

st.divider()

# --- ZONE DE DÉCISION ---
st.subheader("🎯 Votre Stratégie de Prix")

col_prix, col_info = st.columns([1, 2])
with col_prix:
    mon_prix = st.number_input("💸 Fixez votre prix de vente (€)", min_value=cout_revient, max_value=200.0, value=47.5, step=0.5)
    marge_unitaire = mon_prix - cout_revient
    st.caption(f"Votre marge par article : **{marge_unitaire:.2f} €**")

with col_info:
    if mon_prix < prix_moyen_marche:
        st.success(f"💡 Vous êtes **moins cher** que la moyenne ({prix_moyen_marche:.2f} €).")
    else:
        st.warning(f"⚠️ Vous êtes **plus cher** que la moyenne ({prix_moyen_marche:.2f} €).")

# --- BOUTON DE SIMULATION ---
calculer = st.button("📊 Estimer mes gains avec l'IA", type="primary", use_container_width=True)
st.divider()

# ==========================================
# 🧠 CALCULS ET PRÉDICTION
# ==========================================
if calculer:
    if modele_lgbm is not None and scaler is not None:
        try:
            colonnes_modele = modele_lgbm.feature_name_
            colonnes_scaler = scaler.feature_names_in_
            
            X_input_df = pd.DataFrame(0.0, index=[0], columns=colonnes_modele)
            
            mois_dict = {"Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4, "Mai": 5, "Juin": 6,
                         "Juillet": 7, "Août": 8, "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12}
            
            def set_val(col_name, val):
                if col_name in colonnes_modele:
                    X_input_df.at[0, col_name] = val

            # --- Remplissage avec tes saisies ---
            set_val('unit_price', mon_prix)
            set_val('lag_price', prix_precedent)
            set_val('freight_price', frais_port)
            set_val('month', mois_dict[mois_nom])
            set_val('customers', nb_clients)
            set_val('product_score', score_produit)
            set_val('product_photos_qty', nb_photos)
            
            set_val('comp_1', comp_1); set_val('fp1', comp_1)
            set_val('comp_2', comp_2); set_val('fp2', comp_2)
            set_val('comp_3', comp_3); set_val('fp3', comp_3)
            
            # Variables Ingénierie
            avg_comp = (comp_1 + comp_2 + comp_3) / 3
            set_val('avg_comp_price', avg_comp)
            set_val('price_ratio_1', mon_prix / comp_1 if comp_1 > 0 else 1.0)
            set_val('price_ratio_2', mon_prix / comp_2 if comp_2 > 0 else 1.0)
            set_val('price_ratio_3', mon_prix / comp_3 if comp_3 > 0 else 1.0)
            set_val('price_diff_1', mon_prix - comp_1)
            set_val('price_diff_2', mon_prix - comp_2)
            set_val('price_diff_3', mon_prix - comp_3)
            set_val('is_cheaper_than_avg', 1 if mon_prix < avg_comp else 0)
            set_val('customer_score_ratio', nb_clients / score_produit if score_produit > 0 else 0)
            set_val('customer_photo_ratio', nb_clients / nb_photos if nb_photos > 0 else 0)
            
            # --- Remplissage données neutres ---
            set_val('product_weight_g', 1500.0) 
            set_val('product_description_lenght', 500.0)
            set_val('product_name_lenght', 50.0)
            
            # Mise à l'échelle
            X_input_df[colonnes_scaler] = scaler.transform(X_input_df[colonnes_scaler])
            
            # Prédiction brute de l'IA (Sert de note d'attractivité)
            pred_log = modele_lgbm.predict(X_input_df)
            valeur_brute = np.expm1(pred_log)[0]
            
            # --- 🚀 NOUVELLE LOGIQUE RÉALISTE : TRAFIC X CONVERSION X IA ---
            # Taux de conversion e-commerce moyen fixé à 2%
            taux_conversion_moyen = 0.02 
            
            # L'IA module le taux de conversion (plus la valeur brute est haute, plus on convertit)
            conversion_ajustee = taux_conversion_moyen * valeur_brute
            
            # Bonus stratégique : si on est moins cher que la moyenne, la conversion augmente de 20%
            if mon_prix < avg_comp:
                 conversion_ajustee = conversion_ajustee * 1.2
            
            # Calcul du volume final
            quantite_predite = int(round(trafic_estime * conversion_ajustee))
            
            # Sécurité anti-zéro
            if quantite_predite <= 0: quantite_predite = 1
            # -------------------------------------------------------
                
            # Calculs Business
            ca_estime = quantite_predite * mon_prix
            benefice_net = quantite_predite * marge_unitaire
            pourcentage_marge = (marge_unitaire / mon_prix * 100) if mon_prix > 0 else 0
            
            # ==========================================
            # 🖥️ AFFICHAGE DANS LE TERMINAL (CONSOLE)
            # ==========================================
            print("\n" + "="*50)
            print(f"📊 NOUVELLE SIMULATION - Prix de vente : {mon_prix} €")
            print(f"🔹 Trafic estimé : {trafic_estime} visiteurs")
            print(f"🔹 Valeur brute réelle de l'IA (Attractivité) : {valeur_brute:.4f}")
            print(f"🔹 Taux de conversion calculé : {(conversion_ajustee*100):.2f}%")
            print(f"🔹 Quantité finale retenue : {quantite_predite} unités")
            print("="*50 + "\n")

            # ==========================================
            # 🌐 AFFICHAGE WEB
            # ==========================================
            st.caption(f"*(Debug) Conversion calculée par l'IA : {(conversion_ajustee*100):.2f}% pour {trafic_estime} visiteurs.*")
            
            st.subheader("💼 Projections Financières (Estimées par l'IA)")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric(label="📦 Volume de Ventes Prévu", value=f"{quantite_predite} unités")
            with col_res2:
                st.metric(label="💰 Chiffre d'Affaires Brut", value=f"{ca_estime:,.2f} €")
            with col_res3:
                st.metric(label="🏆 Bénéfice Net Estimé", value=f"{benefice_net:,.2f} €", delta=f"{pourcentage_marge:.1f}% de marge")

            if benefice_net > 500:
                st.balloons()
            
            # ==========================================
            # 📖 EXPLICATION PÉDAGOGIQUE EN CLAIR
            # ==========================================
            st.write("") 
            st.markdown("### 💡 Qu'est-ce que cela signifie concrètement ?")
            
            if benefice_net > 0:
                st.info(f"""
                **Traduction des résultats :**
                * Sur la base de vos **{trafic_estime} visiteurs** attendus, l'Intelligence Artificielle a analysé l'attractivité de votre prix ({mon_prix:.2f} €) face aux concurrents.
                * Elle estime que vous allez convertir ces visites en **{quantite_predite} vente(s)** ce mois-ci.
                * Sur chaque article vendu, une fois vos coûts déduits ({cout_revient:.2f} €), il vous reste **{marge_unitaire:.2f} €** de profit net.
                * À la fin du mois, l'ensemble de ces ventes vous rapportera un bénéfice total estimé à **{benefice_net:.2f} €**.
                """)
            elif benefice_net == 0:
                st.warning(f"""
                **Traduction des résultats :**
                * Vous vendez votre produit à prix coûtant (**{mon_prix:.2f} €**). 
                * L'IA estime **{quantite_predite} vente(s)** grâce à votre trafic, mais vous ne gagnez aucun bénéfice net (0 €). 
                * Il s'agit d'une opération blanche. Augmentez votre prix pour générer du profit !
                """)
            else:
                st.error(f"""
                **⚠️ Attention, vous vendez à perte !**
                * Vous avez fixé un prix de **{mon_prix:.2f} €**, ce qui est inférieur à ce que le produit vous coûte ({cout_revient:.2f} €).
                * Chaque fois que vous vendez cet article, vous perdez **{abs(marge_unitaire):.2f} €**.
                * Avec les **{quantite_predite} vente(s)** estimées par l'IA, vous finiriez le mois avec une perte totale de **{abs(benefice_net):.2f} €**.
                """)
                
        except Exception as e:
            st.error(f"🚨 Erreur technique : {e}")
            st.code(e)
    else:
        st.error("Les modèles n'ont pas pu être chargés.")